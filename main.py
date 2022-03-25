import PySimpleGUI as sg  

sg.theme('SystemDefault')

layout = [  [sg.T("This is where the image will go")],
            [sg.Button('Generate'), sg.Button('Cancel')] ]


window = sg.Window('GANS for Animal Faces', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break  
    if event == 'Generate': 
        print("Run GANS")

window.close()


