// Import crates
use aws_sdk_s3::types::AggregatedBytes;
use aws_sdk_s3::{Client, Config, Credentials, Region};

use tch::nn::{ModuleT, OptimizerConfig, Sequential};
use tch::vision::imagenet::load_image_and_resize224_from_memory;
use tch::{nn, Tensor};
//use tch::vision::resnet::resnet34;

// Constants
const AWS_ACCESS_KEY_ENV_VAR: &str = "AWS_ACCESS_KEY_ID";
const AWS_SECRET_KEY_ENV_VAR: &str = "AWS_SECRET_ACCESS_KEY";
const REGION: &str = "us-east-1";
const BUCKET: &str = "pnuemonia-chest-xrays-ids721";

///// Functions /////

// Establish the client
pub fn get_aws_client() -> Client {
    // Establish the environment variables
    let aws_access_key =
        std::env::var(AWS_ACCESS_KEY_ENV_VAR).expect("AWS_ACCESS_KEY_ID not found");
    let aws_secret_key =
        std::env::var(AWS_SECRET_KEY_ENV_VAR).expect("AWS_SECRET_ACCESS_KEY not found");

    // Prep for the client
    let credentials = Credentials::new(
        aws_access_key,
        aws_secret_key,
        None,
        None,
        "loaded-from-env-variables",
    );
    let region = Region::new(REGION);
    let config_builder = Config::builder()
        .region(region)
        .credentials_provider(credentials);
    let config = config_builder.build();

    // Create and return the client
    Client::from_conf(config)
}

// Load the desired data from the s3 bucket
pub async fn load_data(client: Client, object_path: &str) -> Vec<Tensor> {
    // Get the keys from the bucket
    let result = client
        .list_objects_v2()
        .bucket(BUCKET)
        .prefix(object_path)
        .send()
        .await
        .unwrap();
    let objects = result.contents().unwrap();
    let keys = objects
        .iter()
        .map(|object| object.key().unwrap().to_string())
        .collect::<Vec<String>>();

    // Download the data from the bucket
    let mut image_vector = Vec::new();
    for key in keys {
        // Download the object
        let result = client
            .get_object()
            .bucket(BUCKET)
            .key(key.clone())
            .send()
            .await
            .unwrap();
        let data: AggregatedBytes = result.body.collect().await.unwrap();
        let data = data.into_bytes();

        // Convert the bytes into an image
        let image = load_image_and_resize224_from_memory(&data).unwrap();
        //let image = load_image_from_memory(&data).unwrap();
        //let image = image.resize(&[1, 224, 224]);
        image_vector.push(image);
    }
    image_vector
}

// Combine vector images into a single vector of images
pub fn combine_data(
    normal: &mut Vec<Tensor>,
    pneumonia: &mut Vec<Tensor>,
    num_neg_class: usize,
    num_pos_class: usize,
) -> (Vec<Tensor>, Tensor) {
    // Combine the two vectors
    let mut combined = Vec::new();
    combined.append(normal);
    combined.append(pneumonia);

    // generate the labels
    let mut labels = Vec::new();
    for _ in 0..num_neg_class {
        labels.push(0);
    }
    for _ in 0..num_pos_class {
        labels.push(1);
    }
    let labels = Tensor::of_slice(&labels);

    // return the combined vector and associated labels
    (combined, labels)
}

pub fn binary_classifier_nn(vs: &nn::Path) -> Sequential {
    nn::seq()
        .add(nn::conv2d(vs / "conv1", 3, 32, 5, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs / "conv2", 32, 64, 5, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs / "conv3", 64, 128, 5, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add(nn::linear(vs / "fc1", 106, 1024, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "fc2", 1024, 1, Default::default()))
}

// Train the model
pub fn train_model(images: Vec<Tensor>, labels: Tensor, num_epochs: i32) -> Sequential {
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let model = binary_classifier_nn(&vs.root());
    let mut optimizer = nn::Adam::default().build(&vs, 1e-4).unwrap();

    for epoch in 0..num_epochs {
        println!("Epoch: {}", epoch + 1);

        // Loop through the images and labels
        for (i, image) in images.iter().enumerate() {
            // Clear the gradients
            optimizer.zero_grad();

            // Pass the images through the model
            let output = model.forward_t(image, true).sum(tch::Kind::Float);
            let output = output.sigmoid();
            //println!("Output Obtained: {:?}", output);

            // Get the ith label
            let label = labels.get(i.try_into().unwrap());
            //println!("Label Obtained: {:?}", label);

            // Calculate the loss and backpropagate
            let loss = (output - label).abs();
            //println!("Loss Obtained: {:?}", loss);

            loss.backward();
            optimizer.step();
        }
    }
    // Return the model
    model
}

// Test the model
pub fn test_model(model: impl ModuleT, images: Vec<Tensor>, labels: Tensor) -> f64 {
    // Loop through the images and labels
    let mut correct = 0;
    for (i, image) in images.iter().enumerate() {
        // Pass the images through the model
        let output = model.forward_t(image, false).sum(tch::Kind::Float);
        let pred = output.sigmoid();

        // Get the ith label
        let label = labels.get(i.try_into().unwrap());

        // Calculate the prediction
        if pred == label {
            correct += 1;
        }
    }

    // Return the model
    correct as f64 / images.len() as f64
}

// TODO:
// 1. Create a function to save the model
// 2. Create a function to load the model
// 3. Connect with sagemaker
// 4. Train the model for longer

///// DEBUGGING FUNCTIONS /////
pub fn _print_image_sizes(images: &Vec<Tensor>) {
    for image in images {
        println!("Image Shape: {:?}", image.size());
    }
}
