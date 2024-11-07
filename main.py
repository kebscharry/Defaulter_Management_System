from model_training import train_random_forest

def main():
    # Define the path to the Excel file
    file_path = 'activeOnART.xls'

    # Train the model and save it
    train_random_forest(file_path)

if __name__ == '__main__':
    main()
