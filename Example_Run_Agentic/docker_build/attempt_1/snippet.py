import requests

def main():
    """Main program to make a request to the GitHub API."""
    try:
        # Example usage of the requests library
        response = requests.get('https://api.github.com')
        if response.status_code == 200:
            print("Successfully connected to GitHub API!")
            print(f"Response status code: {response.status_code}")
        else:
            print(f"Failed to connect to GitHub API. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Running main program...")
    main()