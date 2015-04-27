
#![feature(step_by)]
/// Protocol Buffers format.
mod protobuf {

    use std::io::{Cursor, Read};

    fn read_uvarint<R: Read>(input: &mut R) -> Option<u32> {
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

fn main() {
    // TODO.
}
