import os
from multiprocessing import Pipe, Process

def send(pipe):
    pipe.send("Hello from the other side")

def talk(pipe):
    pipe.send(dict(name='Bob', spam=42))
    reply = pipe.recv()
    print('reply:', reply)

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p_child = Process(target=send, args=(child_conn,))
    p_parent = Process(target=talk, args=(parent_conn,))
    p_child.start()
    p_parent.start()
    print('parent_conn recv:', parent_conn.recv())   # prints "Hello from the other side"
    print('child recv', child_conn.recv())           # prints "{'name': 'Bob', 'spam': 42}"
    parent_conn.send('Hi from parent')
    print('child recv', child_conn.recv())
    parent_conn.close()
    child_conn.close()