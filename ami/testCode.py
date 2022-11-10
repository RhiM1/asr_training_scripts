import torch


sm = torch.nn.Softmax(dim = 1)

features = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], dtype = torch.float)
ex_features = features.detach().clone()

print(features)
print(ex_features)
print(features.shape)
print(ex_features.shape)


featuresSize = features.size()
# ex_features = ex_features[0:int(round(featuresSize[0] / 2, 0))]

# print("submod features size:", features.size())
# print("submod ex_features size:", ex_features.size())

features = features.view(features.size()[0] * features.size()[1], -1)
ex_features = ex_features.view(ex_features.size()[0] * ex_features.size()[1], -1)

print(features)
print(ex_features)
print(features.shape)
print(ex_features.shape)


# print("submod viewed features size:", features.size())
# print("submod viewed ex_features size:", ex_features.size())

# W = torch.matmul(self.V(features), torch.t(nn.functional.normalize(ex_features, dim = -1)))
W = torch.matmul(features, torch.t(torch.nn.functional.normalize(ex_features, dim = -1)))
W = sm(10000 * W)
print(W)
print(W.shape)

# print("submod W size:", W.size())

A = torch.matmul(W, ex_features)

# print("submod A size:", A.size())

A = A.view(featuresSize)
print(A)
print(A.shape)

# print("submod final A size:", A.size())









