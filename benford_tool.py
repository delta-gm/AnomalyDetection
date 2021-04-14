import numpy as np 
import pandas as pd 
import seaborn as sns
import random
import matplotlib.pyplot as plt
import math

def leadingDigit(n): 
    while n >= 10:  
        n = n / 10
    return int(n) 

BENFORD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]

plt.figure(figsize=(10,8))
plt.plot(BENFORD)
plt.title('Benford law distribution of first digits')
# plt.show()

def chi_square_test(data_count,expected_counts):
    """Return boolean on chi-square test (8 degrees of freedom & P-val=0.05)."""
    chi_square_stat = 0  # chi square test statistic
    for data, expected in zip(data_count,expected_counts):

        chi_square = math.pow(data - expected, 2)

        chi_square_stat += chi_square / expected

    print("\nChi-squared Test Statistic = {:.3f}".format(chi_square_stat))
    print("Critical value at a P-value of 0.05 is 15.51.")    
    return chi_square_stat < 15.51


cc = pd.read_csv('creditcard.csv')
# print(cc.head)

amount_fd = []
amount = cc.Amount.values

for i in amount:
    amount_fd.append(leadingDigit(i))

transaction_fd_counts = pd.Series(amount_fd).value_counts().values
transaction_fd_percent = (transaction_fd_counts/np.sum(transaction_fd_counts))*100

plt.figure(figsize=(10,8))
plt.plot(transaction_fd_percent)
plt.plot(BENFORD)
plt.legend(['Transaction Amounts','BENFORD'])
plt.show()


print(chi_square_test(transaction_fd_percent,BENFORD))

# Since the chi_square test came out to be positive, 
# it means that the above dataset conforms with Benfords law for first digits. 