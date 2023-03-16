use pnuemonia_classification_rs::{
    combine_data, get_aws_client, load_data, test_model, train_model,
};
use tokio;

const NUM_EPOCHS: i32 = 1;

#[tokio::main]
async fn main() {
    // Establish the client
    let client = get_aws_client();

    // Load the data
    println!("Loading the validation data...");
    let mut normal_val = load_data(client.clone(), "chest_xrays/val/NORMAL").await;
    let mut pneumonia_val = load_data(client.clone(), "chest_xrays/val/PNEUMONIA").await;
    let num_normal_val = normal_val.len();
    let num_pneumonia_val = pneumonia_val.len();

    let (val_images, val_labels) = combine_data(
        &mut normal_val,
        &mut pneumonia_val,
        num_normal_val,
        num_pneumonia_val,
    );

    //println!("Loading the test data...");
    let mut normal_test = load_data(client.clone(), "chest_xrays/val/NORMAL").await;
    let mut pneumonia_test = load_data(client.clone(), "chest_xrays/val/PNEUMONIA").await;
    let num_normal_test = normal_test.len();
    let num_pneumonia_test = pneumonia_test.len();

    let (test_images, test_labels) = combine_data(
        &mut normal_test,
        &mut pneumonia_test,
        num_normal_test,
        num_pneumonia_test,
    );

    //pnuemonia_classification_rs::_print_image_sizes(&val_images);

    // Instantiate and train the model
    println!("Training the model...");
    let model = train_model(val_images, val_labels, NUM_EPOCHS);

    // Test the model
    println!("Testing the model...");
    let accuracy = test_model(model, test_images, test_labels);
    println!("Accuracy: {}", accuracy);
}
