import httpx

with httpx.Client() as client:
    try:
        response = client.post("http://httpbin.org/status/503")
        response.raise_for_status()
        print(response.read())
    except httpx.HTTPStatusError as exc:
        print("Caught:", exc)
        print("Reading:", exc.response.read())
