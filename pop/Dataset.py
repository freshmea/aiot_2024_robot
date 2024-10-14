from ipywidgets import widgets
from IPython.display import display, Audio
import pyaudio
import wave, time, os
from threading import Thread
from abc import *
from shutil import copyfile


def Collector(name=None):
    module=None

    if name is None:
        print("Keyward 'name' is requred.\n")
        print("[List]")
        print("Audio : Collect audio datasets as wav file.")
        #print("Camera : Collect picture datasets as jpg file through the camera.")
        return
    elif name.lower() == "audio" or type(name) is _Audio_Collector:
        module = _Audio_Collector()
    else:
        print("Keyward 'name' is wrong.\n")
        print("[List]")
        print("Audio : Collect audio datasets as wav file.")
        #print("Camera : Collect picture datasets as jpg file through the camera.")
        return

    module.show()

class _Audio_Collector:
    def __init__(self, sample_rate=8000):
        self.RATE=sample_rate

        self.rec_btn=widgets.Button(
            description='REC',
            disabled=False,
            button_style='danger',
            tooltip='Record',
            icon='circle',
            layout={'width':'fit-content'}
        )

        self.rec_dur=widgets.BoundedFloatText(
            value=1.0,
            min=0,
            max=60,
            step=0.5,
            description='Duration (sec)',
            disabled=False
        )

        self.rec_prog=widgets.FloatProgress(
            value=1,
            min=0,
            max=1.0,
            step=0.1,
            description='Time',
            bar_style='warning',
            orientation='horizontal'
        )

        self.rec_lab=widgets.Label(value="Ready to record.")

        self.rec_out = widgets.Output(layout={'margin': '10px'})

        self.rec_so_btn=widgets.ToggleButton(
            value=False,
            description='',
            disabled=False,
            button_style='primary',
            tooltip='Save',
            icon='caret-down'
        )

        self.rec_save_out = widgets.Output(layout={'padding':'1em','width':'fit-content','height':'fit-content','border': 'solid 1px #AAA'})

        self.rec_save_path = widgets.Text(
            placeholder='path of datasets',
            value="./audio_datasets",
            description='Path',
            disabled=False
        )

        self.rec_save_lab = widgets.BoundedIntText(
            value=0,
            min=0,
            max=99999999,
            step=1,
            description='Label',
            disabled=False
        )

        self.rec_save_name = widgets.Text(
            placeholder='identifier',
            value="identifier",
            description='Name',
            disabled=False
        )

        self.rec_save_btn=widgets.Button(
            description='Save',
            disabled=False,
            button_style='success',
            tooltip='Save',
            icon='floppy-o',
            layout={'width':'99%'}
        )

        self.rec_save_noti=widgets.Label(value="")

        self.rec_btn.on_click(self._rec_clk)
        self.rec_so_btn.observe(self._so_clk, 'value')
        self.rec_save_btn.on_click(self._save_clk)

        with self.rec_save_out:
            display(self.rec_save_path)
            display(self.rec_save_lab)
            display(self.rec_save_name)
            display(self.rec_save_btn)
            display(self.rec_save_noti)

        with self.rec_out:
            display(Audio([1],rate=self.RATE))

    def _rec(self, duration, CHUNK):
        paud_obj = pyaudio.PyAudio()
        
        wave_obj = wave.open("./._tmp.wav", "w")
        wave_obj.setnchannels(1)
        wave_obj.setframerate(self.RATE)

        stream_obj = paud_obj.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True, frames_per_buffer=CHUNK)
        wave_obj.setsampwidth(paud_obj.get_sample_size(pyaudio.paInt16))

        for _ in range(round(self.RATE / CHUNK * duration)):
            wave_obj.writeframes(stream_obj.read(CHUNK))

        wave_obj.close()
        
        stream_obj.stop_stream()
        stream_obj.close()
        paud_obj.terminate()

    def _rec_clk(self, obj):
        if self.rec_dur.value>=0.1:
            self.rec_prog.bar_style="warning"
            self.rec_out.clear_output(wait=True)
            CHUNK = 128
            RATE = self.RATE
            
            obj.disabled=True
            self.rec_lab.value="Record after 1s."

            dtime=time.time()
            ptime=0
            lesstime=time.time()-dtime
            while lesstime<1.:
                if time.time()-ptime>=0.1:
                    self.rec_prog.value=1-lesstime
                    ptime=time.time()
                lesstime=time.time()-dtime

            self.rec_prog.bar_style="success"
            self.rec_prog.value=0
            self.rec_lab.value="Recording now..."
            obj.button_style='success'
            obj.description="REC..."
            obj.icon='square'

            rec_thread = Thread(target=self._rec, args=(self.rec_dur.value, CHUNK))
            rec_thread.start()
            
            dtime=time.time()
            ptime=0
            lesstime=time.time()-dtime
            while lesstime<self.rec_dur.value:
                if time.time()-ptime>=0.1:
                    self.rec_prog.value=lesstime/self.rec_dur.value
                    ptime=time.time()
                lesstime=time.time()-dtime

            if rec_thread.is_alive():
                rec_thread.join()

            self.rec_prog.value=1.0
            time.sleep(0.1)

            with self.rec_out:
                if os.path.exists("./._tmp.wav"):
                    display(Audio(filename="./._tmp.wav"))
                else:                   
                    display(Audio([1],rate=self.RATE))
                
            obj.disabled=False
            obj.button_style='danger'
            obj.description="REC"
            obj.icon='circle'
            self.rec_lab.value="Ready to record."
            self.rec_prog.value=1.0
            self.rec_prog.bar_style="warning"
        else:
            self.rec_lab.value="Set the duration at least 0.1 sec."

    def _so_clk(self, evt):
        if evt['new'] :
            evt['owner'].icon='caret-up'
            
            self.rec_save_out.clear_output()
            with self.rec_save_out:
                display(self.rec_save_lab)
                display(self.rec_save_name)
                display(self.rec_save_btn)
        else:
            evt['owner'].icon='caret-down'
            self.rec_save_out.clear_output()
            
    def _save_clk(self, obj):
        try:
            if not os.path.exists(self.rec_save_path.value):
                os.mkdir(self.rec_save_path.value)
                
            label = self.rec_save_lab.value
            name = self.rec_save_name.value
            timestamp=time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))+str(int(time.time()*100%100))
            
            filestr = str(label)+"_"+str(name)+"_"+str(timestamp)+".wav"
            
            if os.path.exists("./._tmp.wav"):
                copyfile("./._tmp.wav", self.rec_save_path.value+"/"+filestr)
                self.rec_save_noti.value="Saved."
            else:
                self.rec_save_noti.value="Doesn't exist a recorded."
        except Exception as e:
            if e.errno == 2:
                self.rec_save_noti.value="No such file or directory."
            else:
                self.rec_save_noti.value="An error occured."

    def show(self):
        display(widgets.HBox([
                    widgets.VBox([
                    widgets.HBox([self.rec_prog, self.rec_lab]),
                    widgets.HBox([self.rec_dur, self.rec_btn]),
                    self.rec_out],layout={'padding':'1em','width':'fit-content','height':'fit-content','border': 'solid 1px #AAA'}),
    
                    self.rec_save_out
                    ]))


