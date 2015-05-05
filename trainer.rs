/// Protocol Buffers format.
mod protobuf {
    use std::io::Read;

    /// Reads next UVarint.
    pub fn read_uvarint<R: Read>(input: &mut R) -> Option<u32> {
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
        use std::io::Cursor;

        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x80])), None);
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x00])).unwrap(), 0);
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x03])).unwrap(), 3);
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x8E, 0x02])).unwrap(), 270);
        assert_eq!(read_uvarint(&mut Cursor::new(vec![0x9E, 0xA7, 0x05])).unwrap(), 86942);
    }
}

/// Statistics file reading.
mod stats {
    use std::io::Read;

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
    fn skip_account_header<R: Read>(input: &mut R) -> bool {
        let mut buffer = [0u8; 1];
        input.read(&mut buffer).unwrap() == 1 && input.read(&mut buffer).unwrap() == 1
    }

    #[test]
    fn test_read_account() {
        use std::io::Cursor;

        let account = read_account(&mut Cursor::new(vec![0x3e, 0x3e, 0x03, 0x01, 0x8E, 0x02, 0x9E, 0xA7, 0x05, 0x9D, 0xA7, 0x05])).unwrap();
        assert_eq!(account.id, 3);
        assert_eq!(account.tanks.len(), 1);
        assert_eq!(account.tanks[0].id, 270);
        assert_eq!(account.tanks[0].battles, 86942);
        assert_eq!(account.tanks[0].wins, 86941);
    }
}

/// Collaborative filtering.
mod cf {
    use std::collections::HashMap;

    /// A pair of ID (either account or item) and corresponding rating.
    pub struct Rating {
        pub id: u32,
        pub rating: f32
    }

    /// Vector of `Rating`.
    pub type Ratings = Vec<Rating>;

    /// Maps item ID to `Ratings`.
    pub type RatingTable = HashMap<u32, Ratings>;

    /// Maps a pair of items into their similarity.
    pub type Model = HashMap<(u32, u32), f32>;

    /// Trains model.
    ///
    /// Computes similarity between each pair of items.
    pub fn train(rating_table: RatingTable) -> Model {
        let mut model = Model::new();
        for (i, item_1) in rating_table.keys().enumerate() {
            println!("#{}/{} training", i, rating_table.len());
            for item_2 in rating_table.keys() {
                if item_1 >= item_2 {
                    continue;
                }
                let similarity = pearson(rating_table.get(item_1).unwrap(), rating_table.get(item_2).unwrap());
                model.insert((*item_1, *item_2), similarity);
                model.insert((*item_2, *item_1), similarity);
            }
        }
        println!("Training finished: {} entries in model.", model.len());
        model
    }

    /// Predicts item rating by rated items.
    pub fn predict(model: &Model, rated_items: &Ratings, item: u32) -> f32 {
        let mut similarity_sum = 0.0;
        let mut rating_similarity_sum = 0.0;

        for rated_item in rated_items {
            let similarity = *model.get(&(item, rated_item.id)).unwrap();
            if similarity > 0.0 {
                similarity_sum += similarity;
                rating_similarity_sum += similarity * rated_item.rating;
            }
        }

        rating_similarity_sum / similarity_sum
    }

    fn pearson(ratings_1: &Ratings, ratings_2: &Ratings) -> f32 {
        let rating_map_1 = to_rating_map(ratings_1);
        let rating_map_2 = to_rating_map(ratings_2);

        let mut shared_accounts = Vec::new();
        for account_id in rating_map_1.keys() {
            if rating_map_2.contains_key(account_id) {
                shared_accounts.push(account_id)
            }
        }
        if shared_accounts.len() == 0 {
            return 0.0;
        }

        let mut sum_1 = 0.0;
        let mut sum_2 = 0.0;
        let mut sum_q1 = 0.0;
        let mut sum_q2 = 0.0;
        let mut p_sum = 0.0;

        for account_id in shared_accounts.iter() {
            let rating_1 = *rating_map_1.get(&account_id).unwrap();
            let rating_2 = *rating_map_2.get(&account_id).unwrap();
            sum_1 += rating_1;
            sum_2 += rating_2;
            sum_q1 += rating_1 * rating_1;
            sum_q2 += rating_2 * rating_2;
            p_sum += rating_1 * rating_2;
        }

        let n = shared_accounts.len() as f32;
        let denominator = ((sum_q1 - sum_1 * sum_1 / n) * (sum_q2 - sum_2 * sum_2 / n)).max(0.0).sqrt();

        if denominator < 0.000001 { 0.0 } else { (p_sum - sum_1 * sum_2 / n) / denominator }
    }

    /// Creates a map of account ID to rating.
    fn to_rating_map(ratings: &Ratings) -> HashMap<u32, f32> {
        let mut map = HashMap::new();
        for entry in ratings {
            map.insert(entry.id, entry.rating);
        }
        map
    }

    #[test]
    fn test_train() {
        let mut rating_table = RatingTable::new();
        // Just My Luck
        rating_table.insert(101, vec![
            Rating { id: 1, rating: 3.0 },
            Rating { id: 2, rating: 1.5 },
            Rating { id: 3, rating: 3.0 },
            Rating { id: 4, rating: 2.0 }
        ]);
        // Lady in the Water
        rating_table.insert(102, vec![
            Rating { id: 2, rating: 3.0 },
            Rating { id: 5, rating: 3.0 },
            Rating { id: 3, rating: 2.5 },
            Rating { id: 6, rating: 2.5 },
            Rating { id: 4, rating: 3.0 }
        ]);
        // Snakes on a Plane
        rating_table.insert(103, vec![
            Rating { id: 1, rating: 3.5 },
            Rating { id: 2, rating: 3.5 },
            Rating { id: 5, rating: 4.0 },
            Rating { id: 3, rating: 3.5 },
            Rating { id: 6, rating: 3.0 },
            Rating { id: 4, rating: 4.0 },
            Rating { id: 7, rating: 4.5 }
        ]);
        // Superman Returns
        rating_table.insert(104, vec![
            Rating { id: 1, rating: 4.0 },
            Rating { id: 2, rating: 5.0 },
            Rating { id: 5, rating: 5.0 },
            Rating { id: 3, rating: 3.5 },
            Rating { id: 6, rating: 3.5 },
            Rating { id: 4, rating: 3.0 },
            Rating { id: 7, rating: 4.0 }
        ]);
        // The Night Listener
        rating_table.insert(105, vec![
            Rating { id: 1, rating: 4.5 },
            Rating { id: 2, rating: 3.0 },
            Rating { id: 5, rating: 3.0 },
            Rating { id: 3, rating: 3.0 },
            Rating { id: 6, rating: 4.0 },
            Rating { id: 4, rating: 3.0 }
        ]);
        // You, Me and Dupree
        rating_table.insert(106, vec![
            Rating { id: 1, rating: 2.5 },
            Rating { id: 2, rating: 3.5 },
            Rating { id: 5, rating: 3.5 },
            Rating { id: 3, rating: 2.5 },
            Rating { id: 4, rating: 2.0 },
            Rating { id: 7, rating: 1.0 }
        ]);
        let model = train(rating_table);
        assert!((model.get(&(102, 106)).unwrap() - 0.333333).abs() < 0.000001);
    }

    #[test]
    fn test_predict() {
        let mut model = Model::new();
        model.insert((4, 1), 0.182);
        model.insert((4, 2), 0.103);
        model.insert((4, 3), 0.148);
        let rating = predict(&model, &vec![
            Rating { id: 1, rating: 4.5 },
            Rating { id: 2, rating: 4.0 },
            Rating { id: 3, rating: 1.0 }
        ], 4);
        println!("Predicted rating: {}.", rating);
        assert!(3.184 < rating && rating < 3.185);
    }

    #[test]
    fn test_pearson() {
        let correlation = pearson(
            &vec![
                Rating { id: 1, rating: 2.5 },
                Rating { id: 2, rating: 3.5 },
                Rating { id: 3, rating: 3.0 },
                Rating { id: 4, rating: 3.5 },
                Rating { id: 5, rating: 2.5 },
                Rating { id: 6, rating: 3.0 }
            ],
            &vec![
                Rating { id: 1, rating: 3.0 },
                Rating { id: 2, rating: 3.5 },
                Rating { id: 3, rating: 1.5 },
                Rating { id: 4, rating: 5.0 },
                Rating { id: 5, rating: 3.5 },
                Rating { id: 6, rating: 3.0 }
            ]
        );
        assert!(0.3960 < correlation && correlation < 0.3961);
    }
}

/// CF trainer.
mod trainer {
    use std::io::Read;

    use cf;
    use stats;

    const MIN_BATTLES: u32 = 10;

    /// Inserts account into the ratings table.
    pub fn insert_account(rating_table: &mut cf::RatingTable, account: stats::Account, skip_odd: bool) {
        for (i, tank) in account.tanks.iter().enumerate() {
            if tank.battles < MIN_BATTLES {
                continue;
            }
            if skip_odd && (i % 2 == 1) {
                // This one is for test set.
                continue;
            }
            let entry = cf::Rating {
                id: account.id,
                rating: tank.wins as f32 / tank.battles as f32
            };
            if let Some(entries) = rating_table.get_mut(&tank.id) {
                entries.push(entry);
                continue;
            }
            rating_table.insert(tank.id, vec![entry]);
        }
    }

    /// Reads ratings from the input into hashmap.
    pub fn read_ratings<R: Read>(input: &mut R) -> cf::RatingTable {
        let mut rating_table = cf::RatingTable::new();
        let mut tank_count = 0;
        for i in 1.. {
            if let Some(account) = stats::read_account(input) {
                tank_count += account.tanks.len();
                insert_account(&mut rating_table, account, true);
            } else {
                break;
            }
            if i % 100000 == 0 {
                println!("#{} reading | tanks: {}", i, tank_count);
            }
        }
        rating_table
    }

    pub fn evaluate<R: Read>(input: &mut R, model: cf::Model) -> f32 {
        let mut total_count = 0;
        let mut true_count = 0;
        
        for i in 1.. {
            if let Some(account) = stats::read_account(input) {
                let mut rated_items = cf::Ratings::new();
                let mut account_battles = 0;
                let mut account_wins = 0;

                for (i, tank) in account.tanks.iter().enumerate() {
                    if i % 2 == 0 && tank.battles >= MIN_BATTLES {
                        rated_items.push(cf::Rating { id: tank.id, rating: tank.wins as f32 / tank.battles as f32 });
                        account_battles += tank.battles;
                        account_wins += tank.wins;
                    }
                }

                let account_rating = account_wins as f32 / account_battles as f32;
                for (i, tank) in account.tanks.iter().enumerate() {
                    if i % 2 == 1 && tank.battles >= MIN_BATTLES {
                        let true_rating = tank.wins as f32 / tank.battles as f32;
                        let predicted_rating = cf::predict(&model, &rated_items, tank.id);
                        total_count += 1;
                        if (true_rating > account_rating) == (predicted_rating > account_rating) {
                            true_count += 1;
                        }
                    }
                }
            } else {
                break;
            }
            if i % 100000 == 0 {
                println!(
                    "#{} evaluating | total: {} | true: {} | precision: {}",
                    i, total_count, true_count, 100.0 * true_count as f32 / total_count as f32
                );
            }
        }

        true_count as f32 / total_count as f32
    }

    #[test]
    fn test_insert_account() {
        let mut rating_table = cf::RatingTable::new();
        insert_account(&mut rating_table, stats::Account{ id: 100, tanks: vec![
            stats::Tank { id: 1, battles: 1000, wins: 500 },
            stats::Tank { id: 2, battles: 500, wins: 200 }
        ]}, false);
        insert_account(&mut rating_table, stats::Account{ id: 101, tanks: vec![
            stats::Tank { id: 2, battles: 700, wins: 300 },
            stats::Tank { id: 3, battles: 500, wins: 100 }
        ]}, false);
        insert_account(&mut rating_table, stats::Account{ id: 102, tanks: vec![
            stats::Tank { id: 4, battles: 100, wins: 150 },
            stats::Tank { id: 5, battles: 400, wins: 200 }
        ]}, true);
        assert_eq!(rating_table.len(), 4);
        assert_eq!(rating_table.get(&1).unwrap().len(), 1);
        assert_eq!(rating_table.get(&2).unwrap().len(), 2);
        assert_eq!(rating_table.get(&3).unwrap().len(), 1);
        assert_eq!(rating_table.get(&4).unwrap().len(), 1);
    }

    #[test]
    fn test_read_ratings() {
        use std::io::Cursor;

        let rating_table = read_ratings(&mut Cursor::new(
            vec![0x3e, 0x3e, 0x03, 0x01, 0x8E, 0x02, 0x9E, 0xA7, 0x05, 0x9D, 0xA7, 0x05]));
        assert_eq!(rating_table.get(&270).unwrap()[0].id, 3);
        assert!(rating_table.get(&270).unwrap()[0].rating > 0.999988);
    }
}

fn main() {
    use std::env::args;
    use std::io::{BufReader, Seek, SeekFrom};
    use std::fs::File;
    use std::path::Path;

    let file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    let mut input = BufReader::new(&file);

    let rating_table = trainer::read_ratings(&mut input);
    let model = cf::train(rating_table);
    input.seek(SeekFrom::Start(0)).unwrap();
    let precision = trainer::evaluate(&mut input, model);

    println!("Precision: {}.", 100.0 * precision);
}
