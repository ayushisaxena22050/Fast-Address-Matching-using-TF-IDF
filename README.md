# Fast-Address-Matching-using-TF-IDF

Problem Statement:

Given a dataset of 30 lacs rows which includes name, address,city, state, pincodes. I need to identify people living on same address and make a group of those people.

 Challenges:
 
 1. Address have fuzzy as well as phonetic problem . Like Jaipur is spelt as jaipore, Malviya as Maalviya.
 
 2.Some people have put city twice or thrice while typing the address, which in turn increased the length of my address column.
 
 3.A has address "Plot No-7 xyz colony" and B has address "Plot No-8 xyz colony". Hence, only fuzzy match or any text match algorithm alone can't work.
 
 4. It would be so time consuming to iterate n*n times. 
 
 Solution:
 
 1. Grouped data basis on pincodes and then sort it on the basis of TF-IDF vector.
 
 2. Used sparse matrix and Cosine similarity for faster processing.
 
 3. Extracted Numbers from address and then perform text match to solve challenge 8.
 
