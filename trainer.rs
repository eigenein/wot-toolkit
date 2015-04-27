#![feature(step_by)]
/// Protocol Buffers format.
mod protobuf {
    use std::io::{Cursor, Read};

    /// Reads next UVarint.
    pub fn read_uvarint<R: Read>(input: &mut R) -> Option<u32> {
        let mut value: u32 = 0;

        for shift in (0..).step_by(7) {
            let mut buffer: [u8; 1] = [0];
            if input.read(&mut buffer).unwrap() == 0 {
                return None;
            }
            value |= ((buffer[0] & 0x7F) as u32) << shift;
            if buffer[0] & 0x80 == 0 {
                break;
            }
        }

        Some(value)
    }

    #[test]
    fn test_read_uvarint() {
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x80])), None);
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x00])).unwrap(), 0);
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x03])).unwrap(), 3);
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x8E, 0x02])).unwrap(), 270);
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x9E, 0xA7, 0x05])).unwrap(), 86942);
    }
}

/// Statistics file reading.
mod stats {
    use std::io::{Cursor, Read};

    use protobuf;

    #[derive(Debug)]
    pub struct Tank {
        pub id: u32,
        pub battles: u32,
        pub wins: u32
    }

    #[derive(Debug)]
    pub struct Account {
        pub id: u32,
        pub tanks: Vec<Tank>
    }

    /// Reads next account statistics.
    pub fn read_account<R: Read>(input: &mut R) -> Option<Account> {
        if !check_account_header(input) {
            return None;
        }
        let account_id = protobuf::read_uvarint(input).unwrap();
        let tank_count = protobuf::read_uvarint(input).unwrap();
        let mut tanks = Vec::new();
        for _ in 0..tank_count {
            let tank_id = protobuf::read_uvarint(input).unwrap();
            let battles = protobuf::read_uvarint(input).unwrap();
            let wins = protobuf::read_uvarint(input).unwrap();
            tanks.push(Tank { id: tank_id, battles: battles, wins: wins });
        }
        Some(Account { id: account_id, tanks: tanks })
    }

    /// Checks account header.
    /// TODO: read 2 bytes at once.
    fn check_account_header<R: Read>(input: &mut R) -> bool {
        let mut buffer: [u8; 1] = [0];
        if input.read(&mut buffer).unwrap() != 1 {
            return false; // end of file
        }
        assert_eq!(buffer, [0x3e]);
        assert_eq!(input.read(&mut buffer).unwrap(), 1);
        assert_eq!(buffer, [0x3e]);
        true
    }

    #[test]
    fn test_read_account() {
        let account = read_account(&mut Cursor::new(vec![0x3e, 0x3e, 0x03, 0x01, 0x8E, 0x02, 0x9E, 0xA7, 0x05, 0x9D, 0xA7, 0x05])).unwrap();
        assert_eq!(account.id, 3);
        assert_eq!(account.tanks.len(), 1);
        assert_eq!(account.tanks[0].id, 270);
        assert_eq!(account.tanks[0].battles, 86942);
        assert_eq!(account.tanks[0].wins, 86941);
    }
}

/// Similarity functions.
mod sim {
    fn pearson() {
        // TODO.
    }

    #[test]
    fn test_pearson() {
        // TODO.
    }
}

/// Collaborative filtering.
mod cf {
    // TODO.
}

use std::io;

fn main() {
    println!("Started reading.");

    let mut stdin = io::stdin();
    let mut tank_count = 0;

    for i in 0.. {
        match stats::read_account(&mut stdin) {
            Some(account) => {
                tank_count += account.tanks.len();
            },
            None => { break; }
        }
        if i % 10000 == 0 {
            println!("#{} | tanks: {}", i, tank_count);
        }
    }

    println!("Reading finished.");
}
