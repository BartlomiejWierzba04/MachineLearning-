from sklearn.linear_model import LinearRegression


reg = LinearRegression()


# simple graph of y = 2x 
X = [[1],[2],[3],[4],[5],[6]]
Y = [2,4,6,8,10,12]

reg.fit(X,Y)
test = [[15]]
z = reg.predict(test)

print(z)