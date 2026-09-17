// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::thread;

use pams::cheat_sheet_cascade::{
    service_init, service_query, service_learn, BinaryVector,
};

const SOCKET_PATH: &str = "/tmp/yp_cascade.sock";

fn handle_client(mut stream: UnixStream) {
    let mut header = [0u8; 5]; // 1 byte cmd + 4 bytes count
    loop {
        if stream.read_exact(&mut header).is_err() { break; }
        let cmd = header[0];
        let n = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
        if n == 0 || n > 100_000 { break; }

        let mut data = vec![0u8; n * 64];
        if stream.read_exact(&mut data).is_err() { break; }

        match cmd {
            0x01 => { // QUERY
                let mut cells = Vec::with_capacity(n);
                let mut layers = Vec::with_capacity(n);
                for i in 0..n {
                    let mut v = [0u8; 64];
                    v.copy_from_slice(&data[i * 64..(i + 1) * 64]);
                    let (cell, layer, _) = service_query(&v);
                    cells.push(cell);
                    layers.push(layer);
                }
                let _ = stream.write_all(&(n as u32).to_le_bytes());
                for cell in cells { let _ = stream.write_all(&cell.to_le_bytes()); }
                for layer in layers { let _ = stream.write_all(&(layer as i32).to_le_bytes()); }
            }
            0x02 => { // LEARN
                let mut labels = vec![0u8; n * 8];
                if stream.read_exact(&mut labels).is_err() { break; }
                for i in 0..n {
                    let mut v = [0u8; 64];
                    v.copy_from_slice(&data[i * 64..(i + 1) * 64]);
                    let cell = u64::from_le_bytes([
                        labels[i*8], labels[i*8+1], labels[i*8+2], labels[i*8+3],
                        labels[i*8+4], labels[i*8+5], labels[i*8+6], labels[i*8+7],
                    ]);
                    service_learn(&v, cell);
                }
                let _ = stream.write_all(&[0x00]); // ACK
            }
            _ => { break; }
        }
        let _ = stream.flush();
    }
}

fn main() {
    let _ = std::fs::remove_file(SOCKET_PATH);
    std::fs::create_dir_all("/tmp").ok();
    let listener = UnixListener::bind(SOCKET_PATH).expect("bind failed");
    println!("YP Cascade Service v3");
    println!("Socket: {}", SOCKET_PATH);
    service_init();
    println!("Ready for connections...");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => { thread::spawn(|| handle_client(stream)); }
            Err(e) => { eprintln!("Connection failed: {}", e); }
        }
    }
}
