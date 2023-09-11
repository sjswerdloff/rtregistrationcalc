import os
import sys
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import *
import compute_6dof_from_reg_rtss_plan


class Redirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.insert("end", message)
        self.widget.see("end")


def showWindow():
    root.mainloop()

def open_SROfile_dialog():
    filepath = filedialog.askopenfilename(initialfile=SRO_file_path.get(), title="SRO")
    if (filepath):
        SRO_file_path.set(filepath)

def open_RTSSfile_dialog():
    filepath = filedialog.askopenfilename(initialfile=RTSS_file_path.get(), title="RTSS")
    if (filepath):
        RTSS_file_path.set(filepath)

def open_IonPlanfile_dialog():
    filepath = filedialog.askopenfilename(initialfile=IonPlan_file_path.get(), title="Ion Plan")
    if (filepath):
        IonPlan_file_path.set(filepath)

def calculate():
    if os.path.exists(SRO_file_path.get()) and os.path.exists(RTSS_file_path.get()) and os.path.exists(IonPlan_file_path.get()):
        print("="*30)
        compute_6dof_from_reg_rtss_plan.do_calculate(SRO_file_path.get(), RTSS_file_path.get(), IonPlan_file_path.get())
    else:
        messagebox.showerror("ERROR", "Given DICOM files not found.")


root = Tk()
root.title("RT Registration Calc")
root.geometry("800x600")
root.minsize(640,480)

SRO_file_path = StringVar()
RTSS_file_path = StringVar()
IonPlan_file_path = StringVar()

notebook = ttk.Notebook(root)
notebook.pack(pady=10, fill="both", expand=True)

tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="6 DOF")

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="About")

# TAB1 START
labelSRO = Label(tab1, text="SRO:")
labelSRO.grid(row=0, column=0, padx=5, pady=5, sticky="e")
entrySRO = Entry(tab1, textvariable=SRO_file_path)
entrySRO.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
buttonSRO = Button(tab1, text="...", width=3,command=open_SROfile_dialog)
buttonSRO.grid(row=0, column=2, padx=5, pady=5)

labelRTSS = Label(tab1, text="In-room RTSS:")
labelRTSS.grid(row=1, column=0, padx=5, pady=5, sticky="e")
entryRTSS = Entry(tab1, textvariable=RTSS_file_path)
entryRTSS.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
buttonRTSS = Button(tab1, text="...", width=3, command=open_RTSSfile_dialog)
buttonRTSS.grid(row=1, column=2, padx=5, pady=5)

labelIonPlan = Label(tab1, text="RT Ion Plan:")
labelIonPlan.grid(row=2, column=0, padx=5, pady=5, sticky="e")
entryIonPlan = Entry(tab1, textvariable=IonPlan_file_path)
entryIonPlan.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
buttonIonPlan = Button(tab1, text="...", width=3, command=open_IonPlanfile_dialog)
buttonIonPlan.grid(row=2, column=2, padx=5, pady=5)

buttonStart = Button(tab1, text="Calculate", width=20, command=calculate)
buttonStart.grid(row=3, column=0, columnspan=3, padx=5, pady=10)

textOutput = ScrolledText(tab1)
textOutput.config(background="gray12", foreground="gray88")
textOutput.grid(row=4, columnspan=3, sticky="snew")

tab1.columnconfigure(1, weight=1)
tab1.rowconfigure(4, weight=1)
# TAB1 END

# TAB2 START
textAbout = ScrolledText(tab2)
textAbout.pack(pady=10, fill="both", expand=True)

sys.stdout = Redirector(textOutput)
sys.stderr = Redirector(textOutput)

with open("README.md", "r") as f:
    content = f.read()
    textAbout.insert("end", content)

# TAB2 END

if __name__ == "__main__":
    showWindow()