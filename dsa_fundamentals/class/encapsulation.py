class BankAccount:
    def __init__(self, balance):
        self.__balance = balance
    def deposit(self, amount):
        self.__balance += amount
    def get_balance(self):
        return self.__balance

if __name__ == "__main__":
    account = BankAccount(1000)
    account.deposit(500)
    print(f"Balance: {account.get_balance()}")
    # print(account.__balance)  # Will raise AttributeError