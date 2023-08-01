# print("I'm the first")
# if __name__ == "__main__":
#     print("I'm the second")
#当该文件被直接运行时两行输出

# class People(object):
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#         return
#     def __str__(self):
#         return  self.name + ":" +str(self.age)
#     def __lt__(self, other):
#         return self.name < other.name if self.name != other.name else self.age < other.age
#
# if __name__ == "__main__":
#     print("\t".join([str(item) for item in sorted([People("abc",18),
#                                                    People("abe",19), People("abe",12),People("abc",17)])]))

import torch
a = torch.Tensor([[[1, 1], [2, 2], [3, 3], [4, 4]]])
print(a)
print(a.size())
print(a.size()[0:])
print(a.size()[1:])