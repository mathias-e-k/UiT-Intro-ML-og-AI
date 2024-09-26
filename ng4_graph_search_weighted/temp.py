from heapdict import heapdict

q = heapdict()
q["abc"] = 1
q["123"] = 0
print(q.d)
print(q.popitem())
print("abc" in q)
print(q.popitem())
print("abc" in q)