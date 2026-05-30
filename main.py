from simplerag.rag import rag


def main():
    print(rag("What is the price of the X product?"))
    print(rag("What is the warranty of the Y product?"))
    print(rag("What is the warranty of the Z product?"))


if __name__ == "__main__":
    main()
