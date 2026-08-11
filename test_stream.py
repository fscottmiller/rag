import httpx

with httpx.Client() as client:
    try:
        with client.stream("GET", "https://httpstat.us/503") as response:
            response.raise_for_status()
            print("OK")
    except httpx.HTTPStatusError as exc:
        print("Caught:", exc)
        print("Reading:", exc.response.read())
