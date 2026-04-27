use anyhow::Context;
use embedded_io_adapters::tokio_1::FromTokio;
use futures::future::try_join_all;
use iceoryx2::prelude::*;
use orca_rs::{
    OrcaMotor,
    pdu_payload::{OrcaErrors, OrcaHighSpeedResponsePDU},
};
use std::time::Duration;
use tokio;
use tokio_serial::SerialStream;
mod iox2_message_types;
mod serial_config;
use iox2_message_types::{MotorCommandData, MotorState, MotorStates};

type Port = FromTokio<SerialStream>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut motors: Option<[OrcaMotor<Port>; 9]> = None;

    let node = NodeBuilder::new().create::<ipc::Service>()?;
    let mut position_record: Option<MotorCommandData> = None;

    let command_sub_service = node
        .service_builder(&"orca-motor/position_command_um".try_into()?)
        .publish_subscribe::<MotorCommandData>()
        .open_or_create()?;
    let command_subscriber = command_sub_service.subscriber_builder().create()?;

    println!("Subscriber ready to receive data!");

    let connect_resp_service = node
        .service_builder(&"orca-motor/connect".try_into()?)
        .request_response::<bool, bool>() // request: connect (true) / disconnect (false); response: connected (true) / disconnected (false)
        .open_or_create()?;
    let connect_server = connect_resp_service.server_builder().create()?;
    println!("Request-Response server ready!");

    let state_pub_service = node
        .service_builder(&"orca-motor/state".try_into()?)
        .publish_subscribe::<MotorStates>()
        .open_or_create()?;
    let state_publisher = state_pub_service.publisher_builder().create()?;
    println!("State publisher ready!");

    let motor_error_event_service = node
        .service_builder(&"orca-motor/error_event".try_into()?)
        .event()
        .open_or_create()?;
    let motor_error_notifier = motor_error_event_service.notifier_builder().create()?;

    let motor_config = serial_config::load_config("orca-serial-config.toml", true)?;
    let motor_builders = motor_config.motor.map(|s| {
        tokio_serial::new(s, motor_config.baudrate)
            .parity(tokio_serial::Parity::Even)
            .timeout(std::time::Duration::from_millis(1))
    });

    loop {
        // Handle motor connection request
        while let Some(active_request) = connect_server.receive()? {
            println!(
                "Connection request received! Payload: {}",
                active_request.payload()
            );
            if !active_request.is_connected() {
                println!("Request cancelled!");
            } else {
                // Discard any pending position commands
                while let Some(_) = command_subscriber.receive()? {}

                match (active_request.payload(), motors.as_mut()) {
                    (true, None) => {
                        println!("Opening serial ports...");
                        let new_motors_res: anyhow::Result<Vec<OrcaMotor<Port>>> = motor_builders
                            .iter()
                            .map(|builder| {
                                let port = tokio_serial::SerialStream::open(&builder)
                                    .with_context(|| format!("{:?}", builder))?;
                                let port = FromTokio::new(port);
                                Ok(OrcaMotor::new(port))
                            })
                            .collect();

                            println!("Serial ports opened.");

                        let mut new_motors: [OrcaMotor<Port>; 9] = match new_motors_res {
                            Ok(vec) => match vec.try_into() {
                                Ok(arr) => arr,
                                Err(vec) => {
                                    eprintln!(
                                        "Unexpected number of motors initialized: {} (expected 9)",
                                        vec.len()
                                    );
                                    active_request.loan_uninit()?.write_payload(false).send()?;
                                    continue;
                                }
                            },
                            Err(e) => {
                                eprintln!("Failed to open serial port: {:?}", e);
                                active_request.loan_uninit()?.write_payload(false).send()?;
                                continue;
                            }
                        };
                        
                        println!("Setting sleep mode...");
                        try_join_all(new_motors.iter_mut().map(|motor| {
                            motor.set_mode(orca_rs::register_map::OrcaModeOfOperation::SleepMode)
                        }))
                        .await?;
                        println!("Sleep mode set.");

                        // println!("Setting sleep mode...");
                        // for (i, motor) in new_motors.iter_mut().enumerate() {
                        //     println!("Setting sleep mode for motor {i}...");
                        //     motor
                        //         .set_mode(orca_rs::register_map::OrcaModeOfOperation::SleepMode)
                        //         .await?;
                        //     println!("Sleep mode set for motor {i}.");
                        // }
                        // println!("Sleep mode set.");

                        println!("Motors auto-zeroing...");
                        try_join_all(new_motors.iter_mut().map(|motor| {
                            motor.auto_zero(
                                50,
                                orca_rs::register_map::OrcaAutoZeroExitMode::SleepMode,
                                10,
                                true,
                            )
                        }))
                        .await?;
                        println!("Motors auto-zeroed.");
                        println!("Tuning PID...");
                        try_join_all(
                            new_motors
                                .iter_mut()
                                .map(|motor| motor.tune_pid(800, 200, 100, 500, 300_000)),
                        )
                        .await?;
                        println!("PID tuned.");
                        println!("Enabling high speed...");
                        try_join_all(
                            new_motors
                                .iter_mut()
                                .map(|motor| motor.enable_high_speed(115200, 5)),
                        )
                        .await?;
                        println!("High speed enabled.");

                        motors = Some(new_motors);
                        position_record = None;
                        active_request.loan_uninit()?.write_payload(true).send()?;
                        println!("Motors connected!");
                    }
                    (true, Some(_)) => {
                        position_record = None;
                        active_request.loan_uninit()?.write_payload(true).send()?;
                        println!("Motors were already connected.");
                    }
                    (false, None) => {
                        position_record = None;
                        active_request.loan_uninit()?.write_payload(false).send()?;
                        println!("Motors were already disconnected!");
                    }
                    (false, Some(motor)) => {
                        try_join_all(motor.iter_mut().map(|m| m.disable_high_speed())).await?;
                        try_join_all(motor.iter_mut().map(|m| {
                            m.set_mode(orca_rs::register_map::OrcaModeOfOperation::SleepMode)
                        }))
                        .await?;
                        motors = None;
                        position_record = None;
                        active_request.loan_uninit()?.write_payload(false).send()?;
                        println!("Motors disconnected!");
                    }
                }
            }
        }

        // If motors are connected, process position commands and publish motor states
        if let Some(motors) = motors.as_mut() {
            // Subscriber
            while let Some(sample) = command_subscriber.receive()? {
                position_record = Some(*sample);
            }
            // println!("Current position record: {:?}", position_record);
            // println!(
            //     "{}",
            //     std::time::SystemTime::now()
            //         .duration_since(std::time::UNIX_EPOCH)
            //         .unwrap()
            //         .as_millis()
            // );

            if let Some(position) = position_record {
                let responses: [OrcaHighSpeedResponsePDU; 9] = try_join_all(
                    motors
                        .iter_mut()
                        .zip(position.position_um.iter())
                        .map(|(motor, &pos_um)| motor.send_position_high_speed(pos_um)),
                )
                .await?
                .try_into()
                .ok()
                .unwrap();

                let motor_states: [MotorState; 9] = responses
                    .iter()
                    .map(|resp| match resp {
                        OrcaHighSpeedResponsePDU::Command(payload) => Ok(MotorState {
                            position_um: payload.position_um,
                            force_mn: payload.force_mn,
                            power_w: payload.power_w,
                            temperature_c: payload.temperature_c,
                            voltage_mv: payload.voltage_mv,
                            error: payload.error.into(),
                        }),
                        _ => Err(anyhow::anyhow!(
                            "Unexpected response PDU type received from motor"
                        )),
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .try_into()
                    .ok()
                    .unwrap();

                // Publisher
                state_publisher
                    .loan_uninit()?
                    .write_payload(MotorStates {
                        states: motor_states,
                    })
                    .send()?;

                // If motor error:
                let msgs = motor_states
                    .into_iter()
                    .enumerate()
                    .filter(|(_, x)| x.error != 0)
                    .map(|(i, x)| (i, OrcaErrors::from(x.error)))
                    .collect::<Vec<_>>();
                if msgs.is_empty() == false {
                    eprintln!("Motor errors detected: {:?}", msgs);
                    motor_error_notifier.notify()?;
                }
            }
        } else {
            node.wait(Duration::from_millis(100)).unwrap();
        }
    }
}
