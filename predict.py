
from train import train_model
from utils import estimate_price


def predict():
    """
    Main function to take input from the user and predict the price
    of a car given its mileage"""
    try:
        # Taking input from the user
        mil = float(input("Enter Mileage to predict Price: "))
        if mil < 0:
            raise ValueError("Invlaid input value.")
        price = estimate_price(mil)
        print("Estimated price from trained model: {:.2f}".format(price))
        train_model('./data.csv', mil)

    except Exception as e:
        print(type(e).__name__ + ": " + str(e))
        exit(1)
    except KeyboardInterrupt:
        print("Exiting...")
        exit(1)


if __name__ == "__main__":
    predict()
