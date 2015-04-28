/// Protocol Buffers format.
mod protobuf {
    use std::io;

    /// Reads next UVarint.
    pub fn read_uvarint<R: io::Read>(input: &mut R) -> Option<u32> {
        let mut value: u32 = 0;
        let mut shift: u32 = 0;

        loop {
            let mut buffer = [0u8; 1];
            if input.read(&mut buffer).unwrap() == 0 {
                return None;
            }
            value |= ((buffer[0] & 0x7F) as u32) << shift;
            if buffer[0] & 0x80 == 0 {
                break;
            }
            shift += 7;
        }

        Some(value)
    }

    #[test]
    fn test_read_uvarint() {
        assert_eq!(read_uvarint(&mut io::Cursor::new(vec![0x80])), None);
        assert_eq!(read_uvarint(&mut io::Cursor::new(vec![0x00])).unwrap(), 0);
        assert_eq!(read_uvarint(&mut io::Cursor::new(vec![0x03])).unwrap(), 3);
        assert_eq!(read_uvarint(&mut io::Cursor::new(vec![0x8E, 0x02])).unwrap(), 270);
        assert_eq!(read_uvarint(&mut io::Cursor::new(vec![0x9E, 0xA7, 0x05])).unwrap(), 86942);
    }
}

/// Statistics file reading.
mod stats {
    use std::io;

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
    pub fn read_account<R: io::Read>(input: &mut R) -> Option<Account> {
        if !skip_account_header(input) {
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

    /// Skips account header.
    /// TODO: read 2 bytes at once.
    fn skip_account_header<R: io::Read>(input: &mut R) -> bool {
        let mut buffer = [0u8; 1];
        input.read(&mut buffer).unwrap() == 1 && input.read(&mut buffer).unwrap() == 1
    }

    #[test]
    fn test_read_account() {
        let account = read_account(&mut io::Cursor::new(vec![0x3e, 0x3e, 0x03, 0x01, 0x8E, 0x02, 0x9E, 0xA7, 0x05, 0x9D, 0xA7, 0x05])).unwrap();
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
    use std::collections::HashMap;

    /// Contains account ID and account's rating of the item.
    pub struct Entry {
        pub account_id: u32,
        pub rating: f32
    }

    /// Maps item ID to a vector of `Entry`.
    pub type Ratings = HashMap<u32, Vec<Entry>>;
}

/// CF trainer.
mod trainer {
    use std::io::Read;

    use cf;
    use stats;

    const MIN_BATTLES: u32 = 0;

    /// Inserts account into the ratings table.
    pub fn insert_account(ratings: &mut cf::Ratings, account: stats::Account) {
        for tank in account.tanks {
            if tank.battles < MIN_BATTLES {
                continue;
            }
            let entry = cf::Entry {
                account_id: account.id,
                rating: tank.wins as f32 / tank.battles as f32
            };
            if let Some(entries) = ratings.get_mut(&tank.id) {
                entries.push(entry);
                continue;
            }
            ratings.insert(tank.id, vec![entry]);
        }
    }

    /// Reads ratings from the input into hashmap.
    pub fn read_ratings<R: Read>(input: &mut R, ratings: &mut cf::Ratings) {
        let mut tank_count = 0;
        for i in 1.. {
            if let Some(account) = stats::read_account(input) {
                if i % 2 == 1 {
                    // Skip each second account because of no memory.
                    continue;
                }
                tank_count += account.tanks.len();
                insert_account(ratings, account);
            } else {
                break;
            }
            if i % 100000 == 0 {
                println!("#{} | tanks: {}", i, tank_count);
            }
        }
    }

    #[test]
    fn test_insert_account() {
        let mut ratings = cf::Ratings::new();
        insert_account(&mut ratings, stats::Account{ id: 100, tanks: vec![
            stats::Tank { id: 1, battles: 10, wins: 5 },
            stats::Tank { id: 2, battles: 5, wins: 2}
        ]});
        insert_account(&mut ratings, stats::Account{ id: 101, tanks: vec![
            stats::Tank { id: 2, battles: 7, wins: 3 },
            stats::Tank { id: 3, battles: 50, wins: 1}
        ]});
        assert_eq!(ratings.len(), 3);
        assert_eq!(ratings.get(&1).unwrap().len(), 1);
        assert_eq!(ratings.get(&2).unwrap().len(), 2);
        assert_eq!(ratings.get(&3).unwrap().len(), 1);
    }
}

fn main() {
    use std::env::args;
    use std::io::BufReader;
    use std::fs::File;
    use std::path::Path;

    let file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    let mut input = BufReader::new(&file);

    println!("Started reading.");
    let mut ratings = cf::Ratings::new();
    trainer::read_ratings(&mut input, &mut ratings);
    println!("Reading finished.");
}
