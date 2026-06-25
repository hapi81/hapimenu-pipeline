import runpod

def handler(job):
    return {"status": "ok", "message": "Handler functional!"}

runpod.serverless.start({"handler": handler})
