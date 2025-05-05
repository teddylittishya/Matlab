x1 = randn(100,1) -1
x2 = randn(100,1) +1

k = 5
x1_train = x1(1:50)
x2_train = x2(1:50)
x1_test = x1(51:100)
x2_test = x2(51:100)

error
abs take  (train - test)
sort the vectors
find the class for first k points : difference of x1 and x2
    -   classify each point using first k points
calculate total number of error on train
