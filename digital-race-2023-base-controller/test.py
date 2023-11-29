import queue

q = queue.Queue(5)

q.put(5)
q.put(4)
q.put(35)
q.put(2)

while not q.empty():
    out = q.get()
    print(out)