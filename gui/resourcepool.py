import time
import threading
import datetime

from collections import deque

from PySide6.QtCore import Signal, QObject

from tools.logger import Logger


class DummyConnect(QObject):
    signal = Signal()


class ResourcePool:
    class ResourceMetaData:
        def __init__(self) -> None:
            self.last_access = datetime.datetime.now()
            self.hold = True

        def release(self):
            self.hold = False

        def access(self):
            self.last_access = datetime.datetime.now()
            self.hold = True

        def __str__(self) -> str:
            return f"ResourceMetaData(last_access={self.last_access}, hold={self.hold})"

    class RequireTask:
        def __init__(self, key, load_func=None, callback=None) -> None:
            self.key = key
            self.load_func = load_func
            self.connect = DummyConnect()
            if callback is not None:
                self.connect.signal.connect(callback)

        def invoke(self):
            result = None
            if self.load_func is not None:
                try:
                    result = self.load_func()
                except RuntimeError as e:
                    print("Runtime Error:", e)
                except IOError as ioe:
                    print("IO Error:", ioe)
            return result

    class ReleaseTask:
        def __init__(self, key, grace_period) -> None:
            self.key = key
            self.grace_period = grace_period
            self.create_time = datetime.datetime.now()

    class ResourcePoolProxy(threading.Thread):
        def __init__(self, pool: "ResourcePool") -> None:
            super().__init__()
            self.pool = pool
            self.killed = False

        def start(self):
            self.killed = False
            super().start()

        def stop(self):
            self.killed = True

        def run(self) -> None:
            raise NotImplementedError

    class RequireProxy(ResourcePoolProxy):
        def __init__(self, pool: "ResourcePool", auto_closing=False) -> None:
            super().__init__(pool)
            self.auto_closing = auto_closing

        def run(self) -> None:
            while not self.killed:
                require_task_cleared = False
                task = None
                task_handled = None

                self.pool.lock.acquire()
                require_task_cleared = len(self.pool.require_queue) == 0
                if require_task_cleared and self.auto_closing:
                    self.pool.lock.release()
                    break

                if not require_task_cleared:
                    task = self.pool.require_queue.popleft()
                    task_handled = self.pool.metadata.get(task.key, None)

                # print(f"proxy loop: require_task_num: {len(self.pool.require_queue)}")

                if not require_task_cleared and task is not None:
                    if task_handled is None:
                        self.pool.lock.release()
                        result = task.invoke()
                        self.pool.lock.acquire()
                        if result is not None:
                            self.pool.metadata[
                                task.key
                            ] = ResourcePool.ResourceMetaData()
                            self.pool.pool[task.key] = result
                            self.pool.metadata[task.key].access()
                    else:
                        self.pool.metadata[task.key].access()

                    task.connect.signal.emit()
                self.pool.lock.release()

                if require_task_cleared:
                    time.sleep(0.5)

    class ReleaseProxy(ResourcePoolProxy):
        def __init__(self, pool: "ResourcePool", auto_closing=False) -> None:
            super().__init__(pool)
            self.auto_closing = auto_closing

        def run(self) -> None:
            while not self.killed:
                self.pool.lock.acquire()

                now = datetime.datetime.now()

                continue_pop = True

                while len(self.pool.release_queue) > 0 and continue_pop:
                    task: ResourcePool.ReleaseTask = self.pool.release_queue[0]

                    # print("ReleaseProxy: execute " + task.key)

                    meta_data: ResourcePool.ResourceMetaData = self.pool.metadata.get(
                        task.key, None
                    )

                    if meta_data:
                        if meta_data.hold:
                            self.pool.release_queue.popleft()
                            self.pool.release_task_map.pop(task.key)
                            # print("ReleaseProxy: find on hold " + task.key)
                        else:
                            if task.create_time + task.grace_period < now:
                                self.pool.release_queue.popleft()
                                self.pool.release_task_map.pop(task.key)
                                self.pool.metadata.pop(task.key)
                                self.pool.pool.pop(task.key)
                                # print("ReleaseProxy: erase " + task.key)
                            else:
                                continue_pop = False

                self.pool.lock.release()
                if self.auto_closing:
                    break
                time.sleep(5)

    def __init__(self, auto_closing=False) -> None:
        self.metadata = {}
        self.pool = {}
        self.lock = threading.Lock()

        self.require_queue = deque()

        self.release_task_map = {}
        self.release_queue = deque()

        self.logger = Logger()

        self.require_proxy = ResourcePool.RequireProxy(self, auto_closing)
        self.require_proxy.start()
        self.logger.info("[ResourcePool]Start Require Proxy")

        self.release_proxy = ResourcePool.ReleaseProxy(self, auto_closing)
        self.release_proxy.start()
        self.logger.info("[ResourcePool]Start Release Proxy")

        self.grace_period = datetime.timedelta(seconds=5)

    def require(self, key, runnable=None, callback=None):
        task = ResourcePool.RequireTask(key, runnable, callback)
        self.lock.acquire()
        self.require_queue.append(task)
        self.lock.release()
        return task

    def release(self, key, erase_from_pool=False):
        self.lock.acquire()
        self.require_queue = deque(
            [task for task in self.require_queue if task.key != key]
        )

        if (
            erase_from_pool
            and self.metadata.get(key, None) != None
            and self.release_task_map.get(key, None) == None
        ):
            if self.pool.get(key, None) != None:
                self.metadata[key].release()
                task = ResourcePool.ReleaseTask(key, self.grace_period)
                self.release_task_map[key] = task
                self.release_queue.append(task)
            else:
                self.metadata.pop(key)

        self.lock.release()

    def __del__(self):
        self.terminate_proxy()

    def terminate_proxy(self):
        self.require_proxy.stop()
        self.logger.info("[ResourcePool]Stop Require Proxy.....")
        self.release_proxy.stop()
        self.logger.info("[ResourcePool]Stop Release Proxy.....")
        self.require_proxy.join()
        self.release_proxy.join()
        self.logger.info("[ResourcePool]Terminated")

    def get(self, key):
        result = None
        self.lock.acquire()
        result = self.pool.get(key, None)
        if result is None and self.metadata.get(key, None) != None:
            self.metadata.pop(key)
        self.lock.release()
        return result


if __name__ == "__main__":
    pool = ResourcePool()

    pool.require("a", lambda: print("a"), lambda: print(str(pool.metadata["a"])))
    pool.require("a", lambda: print("a"), lambda: print(str(pool.metadata["a"])))
    pool.require("a", lambda: print("a"), lambda: print(str(pool.metadata["a"])))

    time.sleep(2)

    pool.proxy.stop()
    pool.proxy.join()

    if not pool.proxy.is_alive():
        print("thread killed")
