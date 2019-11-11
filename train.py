from model import VAEModel

input_shape = (32, 32, 3)

def main():
    print("loading model")
    model = VAEModel(input_shape)
    vae = model.load_model(input_shape)
    


if __name__ == "__main__":
    main()