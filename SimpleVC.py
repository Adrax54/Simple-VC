from ttkbootstrap import Style
from tkinter import ttk
from tkinter import Canvas, Toplevel, messagebox, Listbox, simpledialog
from keras import models
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image

import pyaudio
import wave
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import json
import os

style = Style(theme='darkly')

window = style.master
window.geometry('400x500')
window.title("Simple Voice Recognizer")
start_screen = Canvas(window, width=400, height=500)
start_screen.pack()

database = {'Profiles': []}
widgets = []
profiles = {}
predicted = 0

fonter = ttk.Style()
fonter.configure('my.font1', font=('Helvetica', 20))

def record_voice(name, directory):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 5
    filename = "{}.wav".format(name)

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording {}'.format(name))

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording {}'.format(name))

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    path = os.getcwd()
    os.rename("{}/{}".format(path, filename), "{}/Recorded_voices/{}/{}".format(path, directory, filename))

def extract_mel_coef(file):

    librosa_audio, librosa_sample_rate = librosa.load('{}'.format(file))
    mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)

    mfccs_processed = np.mean(mfccs.T, axis=0)

    return mfccs_processed

def Create_Profile(canvas, widgets):

    for object in widgets:
        object.place_forget()

    widgets = []

    name_label = ttk.Label(canvas, text='Please enter your name')
    name_label.place(relx=0.5, rely=0.3, anchor='center')
    widgets.append(name_label)

    new_name = ttk.Entry(canvas)
    new_name.place(relx=0.5, rely=0.4, anchor='center')
    widgets.append(new_name)

    def name_ok():

        name = new_name.get()
        if name in database['Profiles']:
            messagebox.showinfo('Information', 'Profile already exists')
            Start_screen(canvas, widgets)
            pass
        elif name == '':
            messagebox.showinfo('Information', 'No name given')
            Start_screen(canvas, widgets)

        else:
            database['Profiles'].append({name: [{'name': name, 'recordings': [], 'mfcc': []}]})
            path = os.getcwd()
            path1 = path +'/Recorded_voices'
            path2 = path1 + '/{}'.format(name)

            try:
                os.mkdir(path2)
            except:

                messagebox.showinfo('Information', 'Cannot create profile')
                Start_screen(canvas,widgets)
                return

            print(database)
            messagebox.showinfo('Information', 'Profile has been added')
            Start_screen(canvas, widgets)

    name_button = ttk.Button(canvas, text="Create Profile", style='success.TButton', command=lambda: name_ok())
    name_button.place(relx=0.5, rely=0.6, anchor='center')
    widgets.append(name_button)

    pass

def Profile_page(canvas, widgets, name):

    global predicted

    for object in widgets:
        object.place_forget()

    widgets = []

    title = ttk.Label(canvas, text="{}'s profile".format(name))
    title.place(relx=0.8, rely=0.1, anchor='center')
    widgets.append(title)
    frame = ttk.Frame(canvas, borderwidth=1, style='success.TFrame')
    frame.place(relx=0.3, rely=0.7, anchor='center')

    widgets.append(frame)

    scrollbar = ttk.Scrollbar(frame, style='success.TFrame')

    listbox = Listbox(frame, yscrollcommand=scrollbar.set, bg='#212020', fg='white')

    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side='right')
    listbox.pack()

    # Dodawanie do insertboxu już istniejące osoby

    for i in range(len(database['Profiles'])):

        if name in database['Profiles'][i].keys():

            for k in range(len(database['Profiles'][i][name])):

                if len(database['Profiles'][i][name][k]['recordings']) == 0:
                    inserto = database['Profiles'][i][name][k]['name']
                    listbox.insert('end', inserto + '   (No Recording)')

    label_listbox = ttk.Label(canvas, text='Select a person to perform action')
    label_listbox.place(relx=0.3, rely=0.5, anchor='center')
    widgets.append(label_listbox)

    def Add_person():

        flag = False

        person_input = simpledialog.askstring('New Person', "Please enter name", parent=canvas)

        if person_input is not None:

            for i in range(len(database['Profiles'])):

                if name in database['Profiles'][i].keys():
                    database['Profiles'][i][name].append({'name': person_input, 'recordings': [], 'mfcc': []})
                    listbox.insert('end', person_input + '   (No Recording)')
                    print(database)

                    flag = True

            if flag == False:

                messagebox.showinfo('info', 'That person is already added \n'
                                            'if you want to add a person with same name, name him diffrently')

                return

            else:
                pass

        else:

            messagebox.showinfo('info', 'No name was entered')

            return

    def Record():

        value_todo = listbox.get(listbox.curselection())
        mfcc_proccessed = []
        if "(No Recording)" in value_todo:

            value = value_todo.replace("(No Recording)", '')
            value = value.strip()

        else:
            value = value_todo.strip()

        for i in range(len(database['Profiles'])):

            if name in database['Profiles'][i].keys():

                for k in range(len(database['Profiles'][i][name])):

                    if database['Profiles'][i][name][k]['name'] == value:

                        if len(database['Profiles'][i][name][k]['recordings']) > 0:

                            response1 = messagebox.askquestion('recording', 'Voice recording is already present\n'
                                                                            'Would you like to record again?\n'
                                                                            '(Warning, recordidng data will be lost)')
                            if response1 == 'yes':

                                database['Profiles'][i][name][k]['recordings'] = []
                                database['Profiles'][i][name][k]['mfcc'] = []

                            else:

                                return

        messagebox.showinfo('recording', 'Recording will reapeat 5 times, \n say something to the microphone')

        recordings_path = os.getcwd()
        recordings_path = recordings_path + '/Recorded_voices/{}'.format(name)

        for i in range(5):

            response = messagebox.askquestion('recording', 'Start recording number.{} ?'.format(i + 1))
            if response == 'yes':
                record_voice(value + '_' + str(i), name)

                for x in range(len(database['Profiles'])):

                    if name in database['Profiles'][x].keys():

                        for k in range(len(database['Profiles'][x][name])):

                            if database['Profiles'][x][name][k]['name'] == value:
                                database['Profiles'][x][name][k]['recordings'].append('{}_{}.wav'.format(value, i))


            elif response == 'no':

                for i in range(len(database['Profiles'])):

                    if name in database['Profiles'][i].keys():

                        for k in range(len(database['Profiles'][i][name])):

                            if database['Profiles'][i][name][k]['name'] == value:
                                for file in database['Profiles'][i][name][k]['recordings']:
                                    final = recordings_path + '/' + file
                                    print(final)
                                    mfcc = extract_mel_coef(final)
                                    database['Profiles'][i][name][k]['mfcc'].append(mfcc)

                            if len(database['Profiles'][i][name][k]['recordings']) > 0:
                                listbox.delete('anchor')
                                listbox.insert('anchor', value)

                messagebox.showinfo('recording', 'Recording has been stopped')
                return


        for i in range(len(database['Profiles'])):

            if name in database['Profiles'][i].keys():

                for k in range(len(database['Profiles'][i][name])):

                    if database['Profiles'][i][name][k]['name'] == value:
                        for file in database['Profiles'][i][name][k]['recordings']:
                            final = recordings_path + '/' + file
                            print(final)
                            mfcc = extract_mel_coef(final)
                            database['Profiles'][i][name][k]['mfcc'].append(mfcc)

        listbox.delete('anchor')
        listbox.insert('anchor', value)

        return

    def record_sample():

        record_voice('Record_Sample', name)

    def Identification():


        def Collect_names():

            directory = os.getcwd()
            directory = directory + '/Recorded_voices/{}'.format(name)
            names = []
            sep = '_'

            for filename in os.listdir(directory):

                if filename.endswith(".wav"):
                    new_name = filename.split(sep, 1)[0]

                    if filename == 'Record_Sample.wav':
                        pass

                    else:

                        if new_name in names:
                            pass

                        else:

                            names.append(new_name)
            return sorted(names)

        directory = os.getcwd()
        directory = directory + '/Recorded_voices/{}'.format(name)

        persons = Collect_names()
        model = models.load_model('Saved_model')
        try:
            mel_coef = extract_mel_coef(directory + '/Record_Sample.wav')
        except:
            messagebox.showinfo('info', 'No recording sample present \n try recording sample first')
            return

        re = []
        final_features = mel_coef.tolist()
        re.append(final_features)
        prediction = model.predict(re)

        index = np.argmax(prediction)

        messagebox.showinfo('Info', 'Recognized Person: \n {}'.format(persons[index]))

        for i in range(len(database['Profiles'])):

            if name in database['Profiles'][i].keys():

                for k in range(len(database['Profiles'][i][name])):

                    database['Profiles'][i][name][k]['Predicted'] = persons[index]
        predicted = persons[index]
        flag = True


        pass

    def Train_network():

        Dataset = []

        def Collect_data():

            test_data = []
            directory = os.getcwd()
            directory = directory + '/Recorded_voices/{}'.format(name)
            sep = '_'
            for filename in os.listdir(directory):

                if filename.endswith(".wav"):
                    if filename == 'Record_Sample.wav':
                        pass
                    else:
                        mfcc = extract_mel_coef(directory + '/' + filename)
                        name_per = filename.split(sep, 1)[0]
                        test_data.append([mfcc, name_per])

            return test_data

        data = Collect_data()

        Dataset.extend(data)



        pands = pd.DataFrame(Dataset, columns=['feature', 'class_label'])


        X = np.array(pands.feature.tolist())
        y = np.array(pands.class_label.tolist())

        le = LabelEncoder()

        yy = np_utils.to_categorical(le.fit_transform(y))

        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)

        num_labels = yy.shape[1]

        def build_model():

            model = Sequential()
            model.add(Dense(256))
            model.add(Activation('sigmoid'))
            model.add(Dropout(0.5))
            model.add(Dense(256))
            model.add(Activation('sigmoid'))
            model.add(Dropout(0.5))
            model.add(Dense(num_labels))
            model.add(Activation('softmax'))
            model.compile(loss='mse', metrics=['accuracy'])

            return model

        model = build_model()

        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy = 100 * score[1]



        epochs = 100

        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
        score1 = model.evaluate(x_train, y_train, verbose=0)
        score1 = 100 * score1[1]


        score2 = model.evaluate(x_test, y_test, verbose=0)
        score2 = 100 * score2[1]

        messagebox.showinfo('Success', 'Training process completed \n\n'
                                       'Network pre-training accuracy {}\n '
                                       'Network training accuracy : {} \n'
                                       'Network testing accuracy : {}'.format(accuracy, score1, score2))
        model.save('Saved_model')

        pass

    def Show_voice_characteristics(name):

        new_window = Toplevel(canvas)

        for i in range(len(database['Profiles'])):

            if name in database['Profiles'][i].keys():

                for k in range(len(database['Profiles'][i][name])):

                    try:
                        predicted_name = database['Profiles'][i][name][k]['Predicted']
                    except:
                        messagebox.showinfo('information', 'No person was identified\n '
                                                           'First identify a person to show characteristics ')
                        return

            else: return

        directory = os.getcwd()
        directory = directory + '/Recorded_voices/{}'.format(name)

        X, sample_rate = librosa.load(directory + '/' + predicted_name + '_0.wav')
        X1, sample_rate1 = librosa.load(directory + '/Record_Sample.wav')

        mfccs = librosa.feature.mfcc(y = X, sr=sample_rate, n_mfcc=40)
        mfccs1 = librosa.feature.mfcc(y = X1, sr = sample_rate1, n_mfcc=40)

        fig = plt.figure()

        fig.add_subplot(1,2,1)
        librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
        plt.title("{}'s Recorded Voice MFCC".format(predicted_name))

        fig.add_subplot(1,2,2)
        librosa.display.specshow(mfccs1, sr = sample_rate, x_axis='time')
        plt.title("Sampled Voice MFCC")


        fig2 = plt.figure()

        fig2.add_subplot(1,2,1)
        librosa.display.waveplot(X, sr = sample_rate)
        plt.title("{}'s Recorded Voice Soundwave".format(predicted_name))

        fig2.add_subplot(1,2,2)
        librosa.display.waveplot(X1, sr = sample_rate1)
        plt.title("Sampled Voice Soundwave")


        plt.tight_layout()
        canv = FigureCanvasTkAgg(fig, new_window)
        canv.get_tk_widget().pack()
        canv2 = FigureCanvasTkAgg(fig2, new_window)
        canv2.get_tk_widget().pack()



        pass

    def log_out():

        return Start_screen(canvas, widgets)

        pass


    show_voice_characteristic_button = ttk.Button(canvas, text = 'Show voice characteristics',
                                                  style='success.Outline.TButton',
                                                  command = lambda: Show_voice_characteristics(name))

    show_voice_characteristic_button.place(relx=0.7, rely=0.35, anchor='center', width = 150)

    widgets.append(show_voice_characteristic_button)

    sample_button = ttk.Button(canvas,text='Record sample', style='success.Outline.TButton',
                                  command=lambda: record_sample())

    sample_button.place(relx=0.5, rely=0.25, anchor='center', width=300)
    widgets.append(sample_button)

    identify_button = ttk.Button(canvas, text='Identify Voice', style='success.Outline.TButton',
                               command=lambda: Identification())

    identify_button.place(relx=0.3, rely=0.35, anchor='center', width=150)
    widgets.append(identify_button)

    record_button = ttk.Button(canvas, text='Record voice', style='success.Outline.TButton', command=lambda: Record())
    record_button.place(relx=0.70, rely=0.60, anchor='center', width=180)
    widgets.append(record_button)

    add_person = ttk.Button(canvas, text='Add new person', style='success.Outline.TButton',
                            command=lambda: Add_person())
    add_person.place(relx=0.70, rely=0.70, anchor='center', width=180)
    widgets.append(add_person)

    delete_person_button = ttk.Button(canvas, text='Delete person', style='success.Outline.TButton')
    delete_person_button.place(relx=0.70, rely=0.80, anchor='center', width=180)
    widgets.append(delete_person_button)

    log_out_button = ttk.Button(canvas, text='Switch Profile', style='success.Outline.TButton',
                                command=lambda: log_out())
    log_out_button.place(relx=0.25, rely=0.1, anchor='center', )
    widgets.append(log_out_button)

    train_network_button = ttk.Button(canvas, text = 'Train network', style ='success.Outline.TButton',
                                    command=lambda: Train_network())
    train_network_button.place(relx=0.525, rely=0.90, anchor='center', width=300)
    widgets.append(train_network_button)

def Start_screen(canvas, widgets):

    for object in widgets:
        object.place_forget()

    widgets = []
    image1 = Image.open("logo.png")
    image1 = image1.resize((200,100), Image.ANTIALIAS)
    test = ImageTk.PhotoImage(image1)
    label_img = ttk.Label(canvas, image=test)
    label_img.image = test
    label_img.place(relx=0.5, rely=0.25, anchor='center')
    widgets.append(label_img)


    label1 = ttk.Label(canvas, text='Simple VC')
    label1.place(relx=0.5, rely=0.1, anchor='center')
    widgets.append(label1)
    label2 = ttk.Label(canvas, text='Please enter your profile or create a new profile')
    label2.place(relx=0.5, rely=0.4, anchor='center')
    widgets.append(label2)
    profile_entry = ttk.Entry(canvas, text='Enter your name')
    profile_entry.place(relx=0.5, rely=0.5, anchor='center')
    widgets.append(profile_entry)

    def log_in():
        flag = False
        profile_log = profile_entry.get()
        if profile_log != '':
            for i in range(len(database['Profiles'])):
                if profile_log in database['Profiles'][i].keys():
                    Profile_page(canvas, widgets, profile_log)
                    flag = True

        if flag == False:
            print(profile_log)
            messagebox.showinfo('Information', 'Profile doesnt exist')

    def save_json():
        global database

        question = messagebox.askyesno("Notice", "Are you sure you want to save ?\nSaving will overwrite the data",
                                       icon='warning')

        if question:
            with open("saved_database.json", "w") as json_file:
                json.dump(database, json_file)
            messagebox.showinfo('Information', 'Database saved')

        else:
            pass

    def upload_profiles():
        global database

        with open('saved_database.json') as json_file:
            data = json.load(json_file)
        messagebox.showinfo('Information', 'Database loaded')

        database = data

    button1 = ttk.Button(canvas, text="Go to your profile", style='success.TButton', command=lambda: log_in())
    button1.place(relx=0.5, rely=0.6, anchor='center')
    widgets.append(button1)

    button2 = ttk.Button(canvas, text="Create a new profile", style='success.Outline.TButton',
                         command=lambda: Create_Profile(canvas, widgets))
    button2.place(relx=0.5, rely=0.7, anchor='center')
    widgets.append(button2)

    button3 = ttk.Button(canvas, text="Save Profiles", style='success.Outline.TButton', command=lambda: save_json())
    button3.place(relx=0.2, rely=0.1, anchor='center')
    widgets.append(button3)

    button4 = ttk.Button(canvas, text="Load Profiles", style='success.Outline.TButton',
                         command=lambda: upload_profiles())
    button4.place(relx=0.8, rely=0.1, anchor='center')
    widgets.append(button4)

    pass

# RUN
Start_screen(start_screen, widgets)
window.mainloop()
