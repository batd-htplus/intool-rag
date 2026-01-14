"""
Asynchronous background task management using FastAPI BackgroundTasks.
Enables non-blocking document ingestion and embedding.
"""

import asyncio
from typing import Optional, Dict, Any, Callable
from asyncio import Queue
from rag.logging import logger

class AsyncTaskQueue:
    """
    Simple async task queue for background processing.
    Can be replaced with Celery/RQ for production.
    """
    
    def __init__(self, max_workers: int = 4):
        self.queue: Queue = Queue()
        self.max_workers = max_workers
        self.workers = []
        self.running = False
    
    async def start(self):
        """Start background workers"""
        if self.running:
            return
        
        self.running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        
    
    async def stop(self):
        """Stop background workers"""
        self.running = False
        
        await self.queue.join()
        
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
    
    async def _worker(self, worker_id: int):
        """Background worker processing queue"""
        
        while self.running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                try:
                    await task()
                except Exception as e:
                    logger.error(f"Worker {worker_id} task error: {str(e)}")
                finally:
                    self.queue.task_done()
            
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
    
    async def enqueue(self, coro: Callable):
        """Add task to queue"""
        await self.queue.put(coro)
        logger.debug(f"Task enqueued, queue size: {self.queue.qsize()}")
    
    def queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()


_task_queue: Optional[AsyncTaskQueue] = None

def get_task_queue() -> AsyncTaskQueue:
    """Get singleton task queue"""
    global _task_queue
    if _task_queue is None:
        _task_queue = AsyncTaskQueue(max_workers=4)
    return _task_queue


async def startup_background_tasks():
    """Initialize background task system on app startup"""
    queue = get_task_queue()
    await queue.start()


async def shutdown_background_tasks():
    """Clean up background task system on app shutdown"""
    queue = get_task_queue()
    await queue.stop()
