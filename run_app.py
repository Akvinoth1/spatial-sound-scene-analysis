try:
    from app import app
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
except ImportError as e:
    print(f"Error importing app: {e}")
    print("Trying alternative method...")
    try:
        import main
    except ImportError as e:
        print(f"Error importing main: {e}")