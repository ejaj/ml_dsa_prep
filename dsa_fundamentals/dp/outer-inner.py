def outer(name):
    print("Outer function started")
    def inner():
        print(f"👋 Hello, {name}!")
    print("📞 Now calling inner function...")
    inner()  # ← call the inner function
    print("🚪 Outer function ends")
    
outer("kazi")
