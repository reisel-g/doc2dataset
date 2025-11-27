# HTTP microservice client

1. Start the server: `cargo run -p three_dcf_service` (or use the Dockerfile).
2. Install deps: `pip install requests`.
3. Run `python client.py ./datasets/sample.pdf`.
4. The script uploads the PDF via multipart, prints the first chunk of the context text, and dumps the returned metrics JSON.
