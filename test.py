async def dispatcher_loop():
    global seq_counter
    while True:
        async with queue_cv:
            # 等到队列非空
            while not queue_heap:
                await queue_cv.wait()

            # 有队列，先清理过期（只需要清理队首即可，因为我们只允许队首出队）
            neg_p, arrival_ms, seq, item = queue_heap[0]
            now = now_ms()
            if now > item.ddl_ms:
                heapq.heappop(queue_heap)
                item.reject_reason = "deadline_exceeded_in_queue"
                item.start_event.set()
                continue

            # 只尝试队首准入
            admitted, pred, dbg = await admit_or_reject(item.body, item.input_tokens_est)
            if not admitted:
                # 不动队列，等“资源变化/请求结束”再重试
                await queue_cv.wait()
                continue

            # 能准入：出队 + 占用执行槽 + 放行
            heapq.heappop(queue_heap)
            item.admitted_dbg = {"predicted_kv_cache_bytes": pred, **dbg}

        # 注意：不要在持锁期间 acquire sema（避免卡死）
        await sema.acquire()
        item.start_event.set()