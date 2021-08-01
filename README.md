[![Build Status](https://travis-ci.com/lumbytyci/spam-filter.svg?token=WdpMwCGqAwoKjkVtHZ3y&branch=master)](https://travis-ci.com/lumbytyci/spam-filter)

# Detect spam text content using LSTM networks
The gist of this project is to provide a rough stratregy to detect spam text content utilizing LSTM neural networks.
Structure of the said neural network is as follows:<br />
![image](https://user-images.githubusercontent.com/17204788/127767338-e27617a3-c009-4ec6-9854-6b517b85fa96.png)<br />
The goal of this implementation is to provide sufficient accuracy of spam text detection (~0.98 accuracy on the test portion of the data) 

## Reduce spam email traffic by employing proof of work
A simple solution (akin to Hashcash) is provided to limit spam email traffic - by making use of PoW before a client (potential spammer) sends an email.<br />

## Running
```
$ python spam_filter.py [path_to_weights_file]
```
If no weights are supplied, the application enters training mode

## Tests
Run the tests with pytest
