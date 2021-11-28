

class A:
    def __init__(self):
        pass

    def __call__(self, s='Sa'):
        print(f"i am {s}")


if __name__ == "__main__":
    a = A()
    a(s='test')
