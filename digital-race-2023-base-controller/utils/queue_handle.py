import queue

def put_to_queue_no_wait_no_block(item, q):
    
    if q.full():
        try:
            q.get_nowait()
        except:
            pass

    try:
        q.put_nowait(item)
    except:
        pass