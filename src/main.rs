extern crate csv;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use std::error::Error;
use std::fs::File;
use rand::thread_rng;
use rand::sample;
//use rand::seq::SliceRandom;

#[derive(Debug, Deserialize, Serialize)]
struct Record {
    data: Vec<f32>,
    class: String
}

// not complete equality just needed for unit tests
impl PartialEq for Record {
    fn eq(&self, other: &Record) -> bool {
        self.data.get(0) == other.data.get(0)
            && self.class == other.class
    }
}

// loading and handling input
fn load_csv(filename: &str) -> Result<Vec<Record>, Box<Error>> {
    let f = File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(f);

    let records: Vec<Record> = rdr.records().map(|sr|{
        let string_record = sr.unwrap();
        Record{
            data: string_record_to_float_vec(&string_record).unwrap(),
            class: string_record.get(string_record.len()-1).unwrap().to_owned()
        }
    }).collect();
    return Ok(records);
}

fn string_record_to_float_vec(string_record: &csv::StringRecord) -> Result<Vec<f32>, Box<Error>> {
    let results: Vec<f32> = string_record.iter()
        .filter(|x| x.parse::<f32>().is_ok())
        .map(|x| x.parse::<f32>().unwrap()) // safe here as we filter out bad values previously
        .collect();

    return Ok(results);
}

fn records_to_classes(records: &Vec<&Record>) -> Vec<String> {
    records.iter().map(|r| r.class.to_owned()).collect()
}

fn class_to_float(class: &str) -> f32 {
    if class == "R" {
        0.0
    } else {
        1.0
    }
}

// algorithmic code
fn accuracy(actual: Vec<String>, predicted: Vec<&str>) -> f32 {
    let correct =
        actual.iter().zip(predicted.iter()).filter(|&(a, b)| a == b).count();
    correct as f32 / actual.len() as f32 * 100.0
}

// split dataset into k-folds
fn cross_validation_split(dataset: &Vec<Record>, n_folds: i32) -> Vec<Vec<&Record>>  {
    let mut rng = thread_rng();
    let sample_size = dataset.len() as i32 / n_folds;

    let datasets = (1..).take(n_folds as usize).map(|_|{
        //dataset.choose_multiple(&mut rng, sample_size as usize)
        sample(&mut rng, dataset, sample_size as usize)
    }).collect();

    datasets
}

fn perceptron(train: &Vec<Record>, test: &Vec<Record>, l_rate: f32, n_epoch: i32) -> Vec<f32> {
    let weights = train_weights(train, l_rate, n_epoch);
    let predictions = test
        .iter()
        .map(|rec| predict(&rec.data, &weights))
        .collect();

    predictions
}

fn evaluate(dataset: &Vec<Record>, n_folds: i32, l_rate: f32, n_epoch: i32) -> Vec<f32> {
    let folds = cross_validation_split(dataset, n_folds);
    let mut scores = Vec::new();

    for (i,fold) in folds.iter().enumerate() {
        let mut train_set = folds.clone();
        let test_set = folds.clone();
        train_set.remove(i); // remove this fold from the training set

        let ts: Vec<Record> = train_set
            .iter()
            .flat_map(|v| v.iter().map(|x| Record{data: x.data.clone(), class: x.class.to_owned()}))
            .collect();

        let test: Vec<Record> = test_set
            .iter()
            .flat_map(|v| v.iter().map(|x| Record{data: x.data.clone(), class: x.class.to_owned()}))
            .collect();

        let results = perceptron(&ts, &test, l_rate, n_epoch);
        //println!("results: {:?}", results);
        let predictions: Vec<&str> = results.iter().map(|r|{
            if r.to_owned() == 0.0 as f32 {
                "R"
            } else {
                "M"
            }
        }).collect();
        let acc = accuracy(records_to_classes(fold), predictions);
        scores.push(acc);
    }

    return scores;
}

fn predict(row: &Vec<f32>, weights: &Vec<f32>) -> f32 {
    let mut activation = weights[0];
    let size = weights.capacity();

    for i in 0..(size-1) {
        activation += weights[i+1] * row[i];
    }

    if activation >= 0.0 {
        1.0
    } else {
        0.0
    }
}

fn train_weights(train: &Vec<Record>, l_rate: f32, n_epoch: i32) -> Vec<f32> {
    let mut weights = vec![0.0; train[0].data.capacity()+1];
    for epoch in 1..n_epoch {
        let mut sum_err = 0.0;
        for row in train {
            let p = predict(&row.data, &weights);
            let cls = class_to_float(&row.class);
            let mut err = cls - p;
            sum_err += err.powf(2.0);
            weights[0] += l_rate * err;
            for i in 0..(row.data.len()) {
                weights[i + 1] += l_rate * err * row.data[i]
            }
        }
        println!(">epoch: {}, weights: {:?}, lrate: {}, err: {}", epoch, weights, l_rate, sum_err);
    }
    weights
}

fn main() {
    let records = load_csv("sonar.all-data.csv").unwrap();
    let scores = evaluate(&records, 3, 0.01, 500);
    println!("Scores: {:?}", scores);
    let sum: f32 = scores.iter().sum();
    let mean: f32 = sum / scores.len() as f32;
    println!("Accuracy: {:?}", mean);
}


#[cfg(test)]
mod tests {
    use super::*;

    fn records_to_float_vecs(records: &Vec<Record>) -> Vec<Vec<f32>> {
        records.iter().map(|f| f.data.clone()).collect()
    }

    #[test]
    fn test_predict_single_row() {
        let row = vec![2.7810836,2.550537003,0.0];
        let weights = vec![-0.1, 0.20653640140000007, -0.23418117710000003];
        let result = predict(&row, &weights);
        assert_eq!(0.0, result)
    }

    #[test]
    fn test_predict_dataset() {
        let dataset = vec![
            Record{data: vec![2.7810836,2.550537003], class: "R".to_owned()},
            Record{data: vec![1.465489372,2.362125076], class: "R".to_owned()},
            Record{data: vec![3.396561688,4.400293529], class: "R".to_owned()},
            Record{data: vec![1.38807019,1.850220317], class: "R".to_owned()},
            Record{data: vec![3.06407232,3.005305973], class: "R".to_owned()},
            Record{data: vec![7.627531214,2.759262235], class: "M".to_owned()},
            Record{data: vec![5.332441248,2.088626775], class: "M".to_owned()},
            Record{data: vec![6.922596716,1.77106367], class: "M".to_owned()},
            Record{data: vec![8.675418651,-0.242068655], class: "M".to_owned()},
            Record{data: vec![7.673756466,3.508563011], class: "M".to_owned()}
        ];
        let weights = vec![-0.1, 0.20653640140000007, -0.23418117710000003];
        let mut prediction = 0.0;
        for d in dataset {
            prediction = predict(&d.data, &weights);
            println!("{}", format!("Expected={:?}, Predicted={:?}", d.class, prediction));
        }
        assert_eq!(1.0, prediction)
    }

    #[test]
    fn test_train_weights() {
        let dataset = vec![
            Record{data: vec![2.7810836,2.550537003], class: "R".to_owned()},
            Record{data: vec![1.465489372,2.362125076], class: "R".to_owned()},
            Record{data: vec![3.396561688,4.400293529], class: "R".to_owned()},
            Record{data: vec![1.38807019,1.850220317], class: "R".to_owned()},
            Record{data: vec![3.06407232,3.005305973], class: "R".to_owned()},
            Record{data: vec![7.627531214,2.759262235], class: "M".to_owned()},
            Record{data: vec![5.332441248,2.088626775], class: "M".to_owned()},
            Record{data: vec![6.922596716,1.77106367], class: "M".to_owned()},
            Record{data: vec![8.675418651,-0.242068655], class: "M".to_owned()},
            Record{data: vec![7.673756466,3.508563011], class: "M".to_owned()}
        ];

        let weights = train_weights(&dataset, 0.1, 5);
        println!("weights > {:?}", weights);
        assert_eq!([-0.1, 0.20653641, -0.23418123].to_vec(), weights)
    }

    #[test]
    fn test_record_to_float_vec() {
        let string_record = csv::StringRecord::from(vec!["1.0", "2.0", "3.0", "R"]);
        let floats = string_record_to_float_vec(&string_record).unwrap();
        assert_eq!(vec![1.0, 2.0, 3.0], floats)
    }

    #[test]
    fn test_load_csv() {
        let records = load_csv("sonar.all-data.csv").unwrap();
        assert_eq!(records.get(1).unwrap(), &Record{data:vec![0.0262], class:"R".to_owned()});
    }

    #[test]
    fn test_accuracy() {
        let actual1 = vec!["R".to_owned(), "R".to_owned(), "M".to_owned()];
        let predicted1 = vec!["R", "R", "M"];
        assert_eq!(accuracy(actual1, predicted1), 100.0);

        let actual2 = vec!["R".to_owned(), "R".to_owned(), "M".to_owned(), "M".to_owned()];
        let predicted2 = vec!["R", "R", "R", "R"];
        assert_eq!(accuracy(actual2, predicted2), 50.0);

        let actual3 = vec!["R".to_owned(), "R".to_owned(), "M".to_owned(), "M".to_owned()];
        let predicted3 = vec!["M", "M", "R", "R"];
        assert_eq!(accuracy(actual3, predicted3), 0.0);
    }

    #[test]
    fn test_cross_validation_split() {
        let dataset = vec![
            Record{data: vec![2.7810836,2.550537003,0.0], class: "R".to_owned()},
            Record{data: vec![1.465489372,2.362125076,0.0], class: "R".to_owned()},
            Record{data: vec![3.396561688,4.400293529,0.0], class: "R".to_owned()},
            Record{data: vec![1.38807019,1.850220317,0.0], class: "M".to_owned()}
        ];

        cross_validation_split(&dataset, 2);
    }

    #[test]
    fn test_records_to_float_vecs() {
        let dataset = vec![
            Record{data: vec![2.7810836,2.550537003,0.0], class: "R".to_owned()},
            Record{data: vec![1.465489372,2.362125076,0.0], class: "R".to_owned()}
        ];
        let converted = records_to_float_vecs(&dataset);
        //println!("{:?}", converted.get(0).unwrap());
        let first_item = converted.get(0).unwrap().get(0).unwrap();
        let expected_first = 2.7810836 as f32;
        assert_eq!(&expected_first, first_item);

        let second_item = converted.get(0).unwrap().get(1).unwrap();
        let expected_second = 2.550537 as f32;
        assert_eq!(&expected_second, second_item);
    }

    #[test]
    fn test_perceptron() {
        let dataset = vec![
            Record{data: vec![2.7810836,2.550537003], class: "R".to_owned()},
            Record{data: vec![7.673756466,3.508563011], class: "M".to_owned()}
        ];
        let l_rate = 0.01 as f32;
        let n_epoch = 50;
        let results= perceptron(&dataset, &dataset, l_rate, n_epoch);
        println!("{:?}", results)
    }

    #[test]
    fn test_evaluate() {
        let records = load_csv("sonar.all-data.csv").unwrap();
        let scores = evaluate(&records, 3, 0.01, 10);
        println!("Scores: {:?}", scores);
    }
}