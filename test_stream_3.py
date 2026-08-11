import httpx

with httpx.Client() as client:
    try:
        with client.stream("POST", "http://httpbin.org/status/503") as response:
            response.raise_for_status()
            print(response.read())
    except httpx.HTTPStatusError as exc:
        print("Caught:", exc)
        print("Reading:", exc.response.read())
