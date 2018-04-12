import os
import numpy as np
import cv2
import glob
import pickle
import time
import itertools
import json
import math
import PIL
import csv

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from collections import OrderedDict, defaultdict
from scipy.interpolate import UnivariateSpline
from scipy.special import expit

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
scenarioOptions = {
    "results" : "Results.AWS",
    "shape" : (1440, 2560, 3), 
    "current_index" : 20, 
    "step" : 10, 
    "n_frames" : 30, 
}
'''

class VideoMaker:
    def __init__(self, scenarioOptions):
        self.options = scenarioOptions
        self.scenario = []

    def cloneWithSameOptions(self):
        cloneOptions = {k:self.options[k] for k in self.options}
        clone = VideoMaker(cloneOptions)
        return clone

    def append(self, scene):
        self.scenario.append(scene)

    def clear(self):
        self.scenario.clear()

    @property
    def current_index(self):
        return self.options["current_index"]

    @current_index.setter
    def current_index(self, value):
        self.options["current_index"] = value

    @property
    def step(self):
        return self.options["step"]

    @step.setter
    def step(self, value):
        self.options["step"] = value

    @property
    def results(self):
        return self.options["results"]

    @results.setter
    def results(self, value):
        self.options["results"] = value

    @property
    def n_frames(self):
        return self.options["n_frames"]

    def scn_parameters(self):
        return self.options["current_index"], self.options["step"], \
            self.options["results"], self.options["shape"], self.options["n_frames"]

    @staticmethod
    def get_text_positions(np_image, lines, color=None, font_file=None, 
                  font_size=None, vspace=None, start=None, 
                  align=None, valign=None):
    
        cf = np_image.shape[0]//720
        
        if not color: color = (0, 255, 255)
        if not font_file: font_file = "C:/Windows/Fonts/agencyr.ttf"
        if not font_size: font_size = cf*40
        if not vspace: vspace = cf*50
        if not start: start = (cf*70, cf*14)
        if not align: align = "center"
        if not valign: valign = "center"
    
        img_src = np_image*255 if (np_image.dtype == 'float32' and np.max(np_image) <= 1.0) else np_image
        img_pil = Image.fromarray(np.uint8(img_src))
        draw = ImageDraw.Draw(img_pil)
    
        font = ImageFont.truetype(font_file, font_size)
    
        offv = 0
        if valign=="center":
            vs = np_image.shape[0]//vspace
            offv = (vs-len(lines))//2
    
        positions = []
        for i,line in enumerate(lines):
            w, h = draw.textsize(line if isinstance(line,(type(""),)) else str(line), font=font)
            positions.append((start[0] if align=="left" else (np_image.shape[1]-w)//2, start[1]+(i+offv)*vspace))
            
        return positions

    @staticmethod
    def show_text(np_image, lines, color=None, font_file=None, 
                  font_size=None, vspace=None, start=None, 
                  align=None, valign=None):
    
        cf = np_image.shape[0]//720
        
        if not color: color = (0, 255, 255)
        if not font_file: font_file = "C:/Windows/Fonts/agencyr.ttf"
        if not font_size: font_size = cf*40
        if not vspace: vspace = cf*50
        if not start: start = (cf*70, cf*14)
        if not align: align = "center"
        if not valign: valign = "center"
    
        img_src = np_image*255 if (np_image.dtype == 'float32' and np.max(np_image) <= 1.0) else np_image
        img_pil = Image.fromarray(np.uint8(img_src))
        draw = ImageDraw.Draw(img_pil)
    
        font = ImageFont.truetype(font_file, font_size)
    
        offv = 0
        if valign=="center":
            vs = np_image.shape[0]//vspace
            offv = (vs-len(lines))//2
        
        for i,line in enumerate(lines):
            w, h = draw.textsize(line, font=font)
            
            draw.text((start[0] if align=="left" else (np_image.shape[1]-w)//2, start[1]+(i+offv)*vspace), 
                      line, (color), font=font)
            
        img_np = np.asarray(img_pil, dtype=np.uint8)
        return img_np

    @staticmethod
    def scn_text_parameters(scene_options):
        bgcolor = scene_options["bgcolor"] if "bgcolor" in scene_options else (0, 0, 0)
        color = scene_options["color"] if "color" in scene_options else None
        font_file = scene_options["font_file"] if "font_file" in scene_options else None
        font_size = scene_options["font_size"] if "font_size" in scene_options else None
        vspace = scene_options["vspace"] if "vspace" in scene_options else None
        start = scene_options["start"] if "start" in scene_options else None
        align = scene_options["align"] if "align" in scene_options else None
        valign = scene_options["valign"] if "valign" in scene_options else None
        
        return bgcolor, color, font_file, font_size, vspace, start, align, valign
    
    def scn_text_gen(self, text, duration = 1, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
        bgcolor, color, font_file, font_size, vspace, start, align, valign = VideoMaker.scn_text_parameters(scene_options)
    
        img = np.full(shape, bgcolor, dtype=np.uint8)
        img = VideoMaker.show_text(img, text, color=color, font_file=font_file, font_size=font_size, 
                        vspace=vspace, start=start, align=align, valign=valign)
        
        for i in range(int(duration*n_frames)):
            yield ((cix, i), img)
            
    
    def scn_text_bg_gen(self, text, bg, duration = 1, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
        bgcolor, color, font_file, font_size, vspace, start, align, valign = VideoMaker.scn_text_parameters(scene_options)
        
        bg_gen = self.cloneWithSameOptions().scn_process_gen(bg) if isinstance(bg,(list,)) else bg
    
        counter = 0
        while counter < duration*n_frames:
            locus, canvas = next(bg_gen, (None, None))
            if canvas is None: break
    
            img = VideoMaker.show_text(canvas, text, color=color, font_file=font_file, font_size=font_size, 
                            vspace=vspace, start=start, align=align, valign=valign)
       
            yield ((cix, counter), img)
            counter += 1
        
    
    def scn_text_slow_gen(self, text, delay = 3, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
        bgcolor, color, font_file, font_size, vspace, start, align, valign = VideoMaker.scn_text_parameters(scene_options)
        trailer = scene_options["trailer"] if "trailer" in scene_options else 0
        mode = scene_options["mode"] if "mode" in scene_options else "center"
        if mode != "center" and mode != "inplace": mode = "center"
        
        msg = []
        counter = 0
    
        if mode == "center":
            for x in range(len(text)):
                msg.append("")
                for y in range(len(text[x])):
                    msg[x] += text[x][y]
                    counter += 1
            
                    img = np.full(shape, bgcolor, dtype=np.uint8)
                    img = VideoMaker.show_text(img, msg, color=color, font_file=font_file, font_size=font_size, 
                                    vspace=vspace, start=start, align=align, valign=valign)
    
                    for i in range(delay):
                        yield ((cix, counter, i), img)
                    
        elif mode == "inplace":
            img = np.full(shape, bgcolor, dtype=np.uint8)
            positions = VideoMaker.get_text_positions(img, text, font_file=font_file, font_size=font_size, 
                        vspace=vspace, start=start, align=align, valign=valign)
        
            for x in range(len(text)):
                msg.append("")
                if isinstance(text[x],(type(""),)):
                    for y in range(len(text[x])):
                        msg[x] += text[x][y]
                        counter += 1
                
                        img = VideoMaker.show_text(img, [msg[x]], color=color, font_file=font_file, font_size=font_size, 
                                vspace=vspace, start=positions[x], align="left", valign="fixed")
        
                        for i in range(delay):
                            yield ((cix, counter, i), img)
                            
                elif isinstance(text[x],(type(0),)):
                    msg[x] = str(text[x])
                    img = VideoMaker.show_text(img, [msg[x]], color=color, font_file=font_file, font_size=font_size, 
                            vspace=vspace, start=positions[x], align="left", valign="fixed")
        
                    counter += 1
                    for i in range(delay*len(msg[x])):
                        yield ((cix, counter, i), img)
        
                elif isinstance(text[x],(type({}),)):
                    # TODO:
                    pass
        
        counter += 1
        for i in range(int(trailer*n_frames)):
            yield ((cix, counter+i), img)
    
    def scn_image_gen(self, image_file, duration=1, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
            
        img = mpimg.imread(image_file)
        if((img.shape[0] != shape[0]) or img.shape[1] != shape[1]):
            img = cv2.resize(img, (shape[1], shape[0]))
        
        for i in range(int(duration*n_frames)):
            yield ((cix, i), img)
    
    def scn_copy_frames_gen(self, from_folder, from_i, up_to_i, step_i=1, template="frame%04d.jpeg", scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
            
        for i in range(from_i, up_to_i, step_i):
            if os.path.exists(from_folder + "/" + template % i):
                img = mpimg.imread(from_folder + "/" + template % i)
                if((img.shape[0] != shape[0]) or img.shape[1] != shape[1]):
                    img = cv2.resize(img, (shape[1], shape[0]))
                yield ((cix, i), img)
    
    def scn_copy_list_gen(self, from_folder, from_list, template="frame%04d.jpeg", scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
            
        for i, lx in enumerate(from_list):
            img = mpimg.imread(from_folder + "/" + template % lx)
            if((img.shape[0] != shape[0]) or img.shape[1] != shape[1]):
                img = cv2.resize(img, (shape[1], shape[0]))
            yield ((cix, i), img)
            
    def scn_copy_list_with_transitions_gen(self, from_folder, from_list, template="frame%04d.jpeg", delay=10,
                                           scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
        
        prev = None
        for i, lx in enumerate(from_list):
            img = mpimg.imread(from_folder + "/" + template % lx)
            if((img.shape[0] != shape[0]) or img.shape[1] != shape[1]):
                img = cv2.resize(img, (shape[1], shape[0]))
                
            if prev is not None:
                for iq in range(1, delay):
                    result = cv2.addWeighted(img, 1.0*iq/delay, prev, 1.0-1.0*iq/delay, 0)
                    yield ((cix, i-1, iq), result)
            yield ((cix, i, 0), img)
            prev = img
    
    def scn_split_gen(self, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
        
        duration = 10 if "duration" not in scene_options else scene_options["duration"] 
        delay = int(duration*n_frames)
        
        mode = "min" if "mode" not in scene_options else scene_options["mode"]
    
        w = shape[1] # img shape
        h = shape[0]
        sshape = scene_options["shape"] # [[0 1 2] [3 4 5]] or similar
        nw = len(sshape[0])     # say, 3 as in above
        nh = len(sshape)        # say, 2 as in above
        sw = int(w/nw)          # part width
        sh = int(h/nh)          # part height
        
        offsets = [[] for i in range(nh)]
        
        for y in range(nh):
            for x in range(nw):
                offsets[y].append((y*sh, x*sw))
    
        sco_gs = scene_options["generators"]
        generators = [self.cloneWithSameOptions().scn_process_gen(sco_gs[sshape[y][x]]) 
                      if isinstance(sco_gs[sshape[y][x]],(list,)) 
                      else sco_gs[sshape[y][x]]
                      for y in range(nh) for x in range(nw) ]
        
        continueWorking = True
        counter = 0
        while continueWorking:
            canvas = np.zeros(shape, dtype=np.uint8)
            EOFs = []
            for y in range(nh):
                for x in range(nw):
                    g = generators[sshape[y][x]]
                    locus, img = next(g, (None, None))
                    if img is None:
                        EOFs.append(True)
                    else:
                        EOFs.append(False)
                        img_src = img*255 if (img.dtype == 'float32' and np.max(img) <= 1.0) else img
                        rs = cv2.resize(img_src, (sw, sh))
                        offsh = offsets[y][x][0]
                        offsw = offsets[y][x][1]
                        canvas[offsh:offsh+sh,offsw:offsw+sw,:] = rs
    
            #print("EOFs:", EOFs, np.all(EOFs))
            if mode == "min" and np.any(EOFs):
                continueWorking = False
            elif np.all(EOFs):
                continueWorking = False
    
            if continueWorking:
                yield (cix, counter), canvas
                counter += 1
    
            if counter >= delay:
                continueWorking = False
    
    def scn_cut_gen(self, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
    
        duration = 10 if "duration" not in scene_options else scene_options["duration"] 
        delay = int(duration*n_frames)
            
        w = shape[1] # img shape
        h = shape[0]
        sshape = scene_options["shape"] # [[0 1 2] [3 4 5]] or similar
        nw = len(sshape[0])     # say, 3 as in above
        nh = len(sshape)        # say, 2 as in above
        sw = int(w/nw)          # part width
        sh = int(h/nh)          # part height
        
        offsets = [[] for i in range(nh)]
        
        for y in range(nh):
            for x in range(nw):
                offsets[y].append((y*sh, x*sw))
        
        sco_bg = scene_options["bg"]
        bg_gen = self.cloneWithSameOptions().scn_process_gen(sco_bg) if isinstance(sco_bg,(list,)) else sco_bg
                
        sco_gs = scene_options["generators"]
        generators = [self.cloneWithSameOptions().scn_process_gen(sco_gs[sshape[y][x]]) 
                      if (sshape[y][x] in sco_gs and sco_gs[sshape[y][x]] and isinstance(sco_gs[sshape[y][x]],(list,))) 
                      else sco_gs[sshape[y][x]] if (sshape[y][x] in sco_gs and sco_gs[sshape[y][x]])
                      else None
                      for y in range(nh) for x in range(nw) 
                      ]
        
        continueWorking = True
        counter = 0
        while continueWorking:
            locus, canvas = next(bg_gen, (None, None))
            if canvas is None: break
            EOFs = []
            for y in range(nh):
                for x in range(nw):
                    g = generators[sshape[y][x]]
                    if g is None: continue
                    locus, img = next(g, (None, None))
                    if img is None:
                        EOFs.append(True)
                    else:
                        EOFs.append(False)
                        img_src = img*255 if (img.dtype == 'float32' and np.max(img) <= 1.0) else img
                        rs = cv2.resize(img_src, (sw, sh))
                        offsh = offsets[y][x][0]
                        offsw = offsets[y][x][1]
                        canvas = canvas.copy()
                        canvas[offsh:offsh+sh,offsw:offsw+sw,:] = rs
    
            if np.any(EOFs):
                continueWorking = False
                
            if continueWorking:
                yield (cix, counter), canvas
                counter += 1
                
            if counter >= delay:
                continueWorking = False
    
    @staticmethod
    def scn_zoom_parameters(options):
        in_out = options["in_out"] if "in_out" in options else "in"
        delay = options["delay"] if "delay" in options else 20
        corner = options["corner"] if "corner" in options else "tl"
        point = options["point"] if "point" in options else None
        zoom = options["zoom"] if "zoom" in options else 2
        trailer = options["trailer"] if "trailer" in options else 0
        return in_out, delay, corner, point, zoom, trailer
    
    def scn_zoom_gen(self, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
        in_out, delay, corner, _, zoom, trailer = VideoMaker.scn_zoom_parameters(scene_options)
            
        w = shape[1] # img shape
        h = shape[0]
        sw = int(w/zoom)                 # part width
        sh = int(h/zoom)                 # part height
        
        sco_fg = scene_options["fg"]
        g = self.cloneWithSameOptions().scn_process_gen(sco_fg) if isinstance(sco_fg,(list,)) else sco_fg
        
        sco_bg = scene_options["bg"]
        bg_gen = self.cloneWithSameOptions().scn_process_gen(sco_bg) if isinstance(sco_bg,(list,)) else sco_bg
        
        if in_out=="in":
            sizes = zip(range(sw, w+1, (w-sw)//delay), range(sh, h+1, (h-sh)//delay))
        elif in_out=="out":
            sizes = zip(range(w, sw-1, -(w-sw)//delay), range(h, sh-1, -(h-sh)//delay))
        
        for counter, size in enumerate(sizes):
            locus, canvas = next(bg_gen, (None, None))
            if canvas is None: break
                
            locus, img = next(g, (None, None))
            if img is None: break
    
            if corner == "tl":
                offsw, offsh = (0,0)
            elif corner == "br":
                offsw, offsh = w-size[0],size[1]
            elif corner == "tr":
                offsw, offsh = w-size[0],0
            elif corner == "bl":
                offsw, offsh = 0,h-size[1]
                
            img_src = img*255 if (img.dtype == 'float32' and np.max(img) <= 1.0) else img
            rs = cv2.resize(img_src, (size[0], size[1]))
    
            canvas = canvas.copy()
            canvas[offsh:offsh+size[1],offsw:offsw+size[0],:] = rs
    
            yield (cix, counter), canvas
    
        counter += 1
        if canvas is not None:
            for i in range(int(trailer*n_frames)):
                yield ((cix, counter+i), canvas)
    
   
    def scn_zoom_pt_gen(self, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
        
        # delay = in fact, = # of steps 
        in_out, delay, _, point, zoom, trailer = VideoMaker.scn_zoom_parameters(scene_options)
        
        x, y = point
            
        w = shape[1] # img shape
        h = shape[0]
       
        sco_fg = scene_options["fg"]
        g = self.cloneWithSameOptions().scn_process_gen(sco_fg) if isinstance(sco_fg,(list,)) else sco_fg
        
        sco_bg = scene_options["bg"]
        bg_gen = self.cloneWithSameOptions().scn_process_gen(sco_bg) if isinstance(sco_bg,(list,)) else sco_bg
        
        # bounding box corners are on the lines: (0,0) to (x, y) and on (x, y) to (w, h)
        # line to tl corner: a*(x,y)  a: [0,1]
        # line to br corner (w, h)+a*((x,y)-(w, h)) a: [0,1]
        # a for zoom out - not sure about uniform, but let's try: 
        a = expit(10*(np.arange(0.0, 1.0, 1./delay)-0.5))
    
        tls = list(zip(np.int_(a*x), np.int_(a*y)))
        brs = list(zip(np.int_(w+a*(x-w)), np.int_(h+a*(y-h))))
        tls.append((x, y))
        tls.insert(0, (0, 0))
        brs.append((x, y))
        brs.insert(0, (w, h))
        
        if in_out=="in":
            bboxes = zip(reversed(list(tls)), reversed(list(brs)))
        elif in_out=="out":
            bboxes = zip(tls, brs)
        
        for counter, bbox in enumerate(bboxes):
            locus, canvas = next(bg_gen, (None, None))
            if canvas is None: break
                
            locus, img = next(g, (None, None))
            if img is None: break
                
            tl, br = bbox
            size = br[0]-tl[0], br[1]-tl[1]
            offsw, offsh = tl
    
            canvas = canvas.copy()
    
            if size != (0, 0):
                img_src = img*255 if (img.dtype == 'float32' and np.max(img) <= 1.0) else img
                rs = cv2.resize(img_src, (size[0], size[1]))
                canvas[offsh:offsh+size[1],offsw:offsw+size[0],:] = rs
    
            yield (cix, counter), canvas
    
        counter += 1
        if canvas is not None:
            for i in range(int(trailer*n_frames)):
                yield ((cix, counter+i), canvas)
    
    
    def scn_draw_gen(self, scene_options = {}):
        cix, step, results, shape, n_frames = self.scn_parameters()
    
        sco_bg = scene_options["bg"]
        bg_gen = self.cloneWithSameOptions().scn_process_gen(sco_bg) if isinstance(sco_bg,(list,)) else sco_bg
    
        operations = scene_options["operations"]
        
        counter = 0
        while True:
            locus, canvas = next(bg_gen, (None, None))
            if canvas is None: break
                
            canvas = canvas.copy()
    
            for operation in operations:
                if operation["cmd"] == "circle":
                    cv2.circle(canvas, *operation["args"])
                elif operation["cmd"] == "ellipse":
                    cv2.ellipse(canvas, *operation["args"])
                elif operation["cmd"] == "fillConvexPoly":
                    cv2.fillConvexPoly(canvas, *operation["args"])
                elif operation["cmd"] == "fillPoly":
                    cv2.fillPoly(canvas, *operation["args"])
                elif operation["cmd"] == "line":
                    cv2.line(canvas, *operation["args"])
                elif operation["cmd"] == "arrowedLine":
                    cv2.arrowedLine(canvas, *operation["args"])
                elif operation["cmd"] == "rectangle":
                    cv2.rectangle(canvas, *operation["args"])
                elif operation["cmd"] == "polylines":
                    cv2.polylines(canvas, *operation["args"])
    
            yield (cix, counter), canvas
            counter += 1
    
    def scn_loop_gen(self, scenario, iterations=1):
        for i in range(iterations):
            for locus, img in self.scn_process_gen(scenario):
                yield locus, img
    
    def scn_process_gen(self, scenario):
        for scene in scenario:

            if "description" not in scene:
                print(self.options["current_index"], "Description is missing.")
            else:
                print(self.options["current_index"], scene["description"])

            scene_options = scene["options"] if "options" in scene else {}
            
            if scene["type"] == "text":
                for locus, img in self.scn_text_gen(scene["text"], scene["duration"], scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]
    
            elif scene["type"] == "text_bg":
                for locus, img in self.scn_text_bg_gen(scene["text"], scene["bg"], scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]

            elif scene["type"] == "text_slow":
                for locus, img in self.scn_text_slow_gen(scene["text"], scene["delay"], scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]
     
            elif scene["type"] == "image":
                for locus, img in self.scn_image_gen(scene["image"], scene["duration"], scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]
                
            elif scene["type"] == "copy_frames":
                step_i = scene["step_i"] if "step_i" in scene else 1
                if "template" in scene:
                    for locus, img in self.scn_copy_frames_gen(scene["from_folder"], scene["from_i"], 
                        scene["up_to_i"], template=scene["template"], step_i=step_i, scene_options=scene_options):
                        yield locus, img
                else:
                    for locus, img in self.scn_copy_frames_gen(scene["from_folder"], scene["from_i"], 
                        scene["up_to_i"], step_i=step_i, scene_options=scene_options):
                        yield locus, img
                self.options["current_index"] += self.options["step"]
                        
            elif scene["type"] == "copy_list":
                if "template" in scene:
                    for locus, img in self.scn_copy_list_gen(scene["from_folder"], scene["from_list"],
                        template=scene["template"], scene_options=scene_options):
                        yield locus, img
                else:
                    for locus, img in self.scn_copy_list_gen(scene["from_folder"], scene["from_list"],
                        scene_options=scene_options):
                        yield locus, img
                self.options["current_index"] += self.options["step"]
                        
            elif scene["type"] == "copy_list_with_transitions":
                if "template" in scene:
                    for locus, img in self.scn_copy_list_with_transitions_gen(scene["from_folder"], 
                        scene["from_list"], template=scene["template"], delay=scene["delay"], scene_options=scene_options):
                        yield locus, img
                else:
                    for locus, img in self.scn_copy_list_with_transitions_gen(scene["from_folder"], 
                        scene["from_list"], delay=scene["delay"],scene_options=scene_options):
                        yield locus, img
                self.options["current_index"] += self.options["step"]
     
            elif scene["type"] == "split":
                for locus, img in self.scn_split_gen(scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]
     
            elif scene["type"] == "cut":
                for locus, img in self.scn_cut_gen(scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]
     
            elif scene["type"] == "zoom":
                for locus, img in self.scn_zoom_gen(scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]
     
            elif scene["type"] == "zoom_pt":
                for locus, img in self.scn_zoom_pt_gen(scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]
     
            elif scene["type"] == "draw":
                for locus, img in self.scn_draw_gen(scene_options=scene_options):
                    yield locus, img
                self.options["current_index"] += self.options["step"]
                
            elif scene["type"] == "loop":
                for locus, img in self.scn_loop_gen(scene["scenario"], scene["iterations"]):
                    yield locus, img
                
            elif scene["type"] == "scenario":
                for locus, img in self.scn_process_gen(scene["scenario"]):
                    yield locus, img

        
    def process(self):
        for locus, img in self.scn_process_gen(self.scenario):
            template = "/frame" + "%04d."*len(locus) + "jpeg" 
            mpimg.imsave(self.options["results"] + template % locus, img)
