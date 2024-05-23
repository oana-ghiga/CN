# import sys
# print(sys.float_info.epsilon)
import tkinter as tk

def precision():
    m = 1
    u = 10**(-m)
    while 1+u != 1:
        prev_m = m
        m += 1
        prev_u = u
        u = 10**(-m)
    result_label.config(text=f"u: {prev_u}, m: {prev_m}")

root = tk.Tk()

calculate_button = tk.Button(root, text="Run!", command=precision)
calculate_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()