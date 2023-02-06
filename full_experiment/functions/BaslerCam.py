# created 25/08/2020 at 13:30
# author: Kira
# wrapper for basler camera

import pypylon
import numpy as np

class BaslerCamera():
  
  def __init__(self, serial,exptime=0.1,gain=1,bins=1):
    self._serial  = serial
    self._exptime = exptime *1e6 # in seconds
    self._gain    = gain
    self._bins    = bins
    self._cam     = None
    
    available_cameras = pypylon.factory.find_devices()
    #print('Available devices:',available_cameras)
    for device in available_cameras:
        #print(type(device.serial_number))
        if device.serial_number==str(self._serial):
            self._cam = pypylon.factory.create_device(device)
            self._cam.open()
            print("Opened camera with serial number:",self._serial)
    if self._cam == None:
        print("Device with serial:",self._serial," not found")
    
    self._cam.properties['PixelFormat'] = 'Mono12'
    self._cam.properties['GainAuto'] = 'Off'
    self._cam.properties['Gain'] = self._gain
    
    self._cam.properties['OffsetY'] = 0
    self._cam.properties['OffsetX'] = 0
    self._cam.properties['Height'] = self._cam.properties["HeightMax"]
    self._cam.properties['Width'] = self._cam.properties["WidthMax"]
    self._cam.properties['BinningHorizontal'] = self._bins
    self._cam.properties['BinningVertical'] = self._bins
            
    self._cam.properties['ExposureMode'] = 'Timed'
    self._cam.properties['ExposureAuto'] = 'Off'
    self._cam.properties['ExposureTime'] = self._exptime
    
    print('Camera info of camera object:', self._cam.device_info)
    
  def takeImage(self,exptime=0):
    if exptime!=0: self._exptime = exptime*1e6
    self._cam.properties['ExposureTime'] = int(self._exptime)
    self._image = (self._cam.grab_image()).astype(float)
  def getImage(self,exptime=0):
    if exptime!=0: self._exptime = exptime*1e6
    self._cam.properties['ExposureTime'] = self._exptime
    self._image = (self._cam.grab_image()).astype(np.int16)   
    return np.flipud(self._image)   # fix up-down flipped image
  def getStoredImage(self):
    return self._image
  def saveImage(self,path):
    np.savez_compressed(path,self._image)
    print('Image saved to:',path)
    
    
  def getPropterties(self):
      return self._cam.properties.keys()
  def closeCam(self):
    self._cam.close()
    print("Closed:",self._cam.device_info)

  