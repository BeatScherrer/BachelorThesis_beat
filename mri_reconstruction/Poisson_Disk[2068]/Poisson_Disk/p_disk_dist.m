function varargout = p_disk_dist(varargin)
% P_DISK_DIST MATLAB code for p_disk_dist.fig
%      P_DISK_DIST, by itself, creates a new P_DISK_DIST or raises the existing
%      singleton*.
%
%      H = P_DISK_DIST returns the handle to a new P_DISK_DIST or the handle to
%      the existing singleton*.
%
%      P_DISK_DIST('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in P_DISK_DIST.M with the given input arguments.
%
%      P_DISK_DIST('Property','Value',...) creates a new P_DISK_DIST or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before p_disk_dist_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to p_disk_dist_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help p_disk_dist

% Last Modified by GUIDE v2.5 15-Feb-2013 14:16:41

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @p_disk_dist_OpeningFcn, ...
                   'gui_OutputFcn',  @p_disk_dist_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before p_disk_dist is made visible.
function p_disk_dist_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to p_disk_dist (see VARARGIN)

% Choose default command line output for p_disk_dist
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes p_disk_dist wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = p_disk_dist_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% Set default values
% Lines and partitions
handles.nl = str2double(get(handles.nL,'String')) ;
handles.np = str2double(get(handles.nP,'String')) ;
% Uniform or Gaussian
handles.wu = get(handles.wU, 'Value') ;
handles.ng = str2double(get(handles.nG,'String')) ;
% Offset, exp tail, slope
handles.wo = get(handles.wO, 'Value') ;
handles.no = str2double(get(handles.nO,'String')) ;
handles.wt = get(handles.wT, 'Value') ;
handles.ns = str2double(get(handles.nS,'String')) ;
handles.ws = get(handles.wS, 'Value') ;
handles.os = str2double(get(handles.oS,'String')) ;
% Identical in x and y, x factor, 3D plot
handles.wi = get(handles.wI, 'Value') ;
handles.wx = get(handles.wX, 'Value') ;
handles.nx = str2double(get(handles.nX,'String')) ;
handles.wp = get(handles.wP, 'Value') ;
% Stepsize and neighbors
handles.nss = str2double(get(handles.nSS,'String')) ;
handles.npp = str2double(get(handles.nPP,'String')) ;
% k-space center
handles.wc = get(handles.wC, 'Value') ;
handles.ncl = str2double(get(handles.nCL,'String')) ;
handles.ncp = str2double(get(handles.nCP,'String')) ;
% Output options
handles.tp = get(handles.tP, 'Value') ;
handles.ta = get(handles.tA, 'Value') ;
handles.tk = get(handles.tK, 'Value') ;
handles.pd_1d = get(handles.pD_1d, 'Value') ;
handles.pd_3d = get(handles.pD_3d, 'Value') ;


handles.tkp = get(handles.tKP, 'Value') ;
handles.ttp = get(handles.tTP, 'Value') ;

% Initiate other handles
handles.patplot = 0 ;
handles.ha = 1-handles.no ;
handles.hb = 0 ;
handles.a = 0 ;
handles.b = 0 ;
handles.d = 0 ;
handles.osv = handles.os ;
% Add file names?

% Update handles structure
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function nL_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nL (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nP_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nG_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nO_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nO (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nS_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nX_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nX (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nSS_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nSS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nPP_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nPP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nCL_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nCL (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function nCP_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nCP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function tPD_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tPD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function tAD_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tAD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% ************************************************************
% *********************************************************
function int_Callback(hObject, eventdata, handles)
% hObject    handle to integers 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(hObject,'BackgroundColor','white');
h = get(hObject, 'UserData') ;
num = str2double(get(hObject,'String')) ;

% Check for integer
if isnumeric(num) == 0
    errordlg('Input an integer', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end
% Ensure it is a number
if isnan(num)
    errordlg('Input an integer', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end
if num <= 0
    errordlg('Input a positive integer', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end
% ************************************************************
% Values for k-space (and fully sampled center) must be even
if   strcmp(h,'nl') == 1 || strcmp(h,'np') == 1 ...
        || strcmp(h,'ncl') == 1 || strcmp(h,'ncp') == 1
    if mod(num,2) ~= 0
        errordlg('Input must be even integer', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
    end
end

% *************************************************************
% Other Values
if   strcmp(h, 'ncl') == 1 && num > handles.nl ...
        || strcmp(h, 'ncp') == 1 && num > handles.np 
        errordlg('Input exceeds k-space', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end
if   strcmp(h, 'npp') == 1 && num > 40
    errordlg('Input may be too large', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end

% Update handles structure
han_str = ['handles.' h, '= num', ';'] ;
eval(han_str) ;
guidata(hObject, handles) ;

% ***************************************************************
% ******************************************************************

% ************************************************************
% *********************************************************
function num_Callback(hObject, eventdata, handles)
% hObject    handle to integers 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(hObject,'BackgroundColor','white');
h = get(hObject, 'UserData') ;
num = str2double(get(hObject,'String')) ;

% Ensure it is a number greater than 0
if isnumeric(num) == 0
    errordlg('Input a number', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end
if num <= 0
    errordlg('Input a positive number', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end
% ************************************************************

% Values for many parameters must be less than 1
if   strcmp(h, 'ng') == 1 || strcmp(h,'no') == 1 ...
        || strcmp(h, 'ns') == 1 || strcmp(h,'nx') == 1 
    if num > 1
        errordlg('Input must be less than 1', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
    end
end

if   strcmp(h, 'nss') == 1 && num < 0.7
    errordlg('Input may be too small', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end

% *****************************************************
% Update handles structure
han_str = ['handles.' h, '= num', ';'] ;
eval(han_str) ;
guidata(hObject, handles) ;
% ***************************************************************
% ******************************************************************

% ************************************************************
% *********************************************************
function cbox_Callback(hObject, eventdata, handles)
% hObject    handle to integers 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

h = get(hObject, 'UserData'); 
num = (get(hObject,'Value')); 

% If uniform pattern turned on, turn off Gaussian, etc
if strcmp(h,'wu') == 1 
    if num == 1
        handles.wg = 0 ;
        set(handles.wG,'Value',0) ;
        set(handles.wO,'Value',0) ;
        set(handles.wT,'Value',0) ;
        set(handles.wS,'Value',0) ;
        set(handles.wX,'Value',0) ;
        set(handles.wP,'Value',0) ;
        set(handles.nG,'Visible','off') ;
        set(handles.nO,'Visible','off') ;
        set(handles.nS,'Visible','off') ;
        set(handles.oS,'Visible','off') ;
        set(handles.nX,'Visible','off') ;
    else
        handles.wg = 1 ;
        set(handles.wG,'Value',1) ;
        set(handles.wO,'Value',0) ;
        set(handles.wT,'Value',0) ;
        set(handles.wS,'Value',0) ;
        set(handles.wX,'Value',0) ;
        set(handles.wP,'Value',0) ;
        set(handles.nG,'Visible','on') ;
        set(handles.nO,'Visible','off') ;
        set(handles.nS,'Visible','off') ;
        set(handles.nS,'Visible','off') ;
        set(handles.nX,'Visible','off') ; 
    end
end

% If Gaussian selected, turn uniform off, etc
if strcmp(h,'wg') == 1
    if num == 1 
        handles.wu = 0 ;
        set(handles.wU,'Value',0) ;
        set(handles.text8,'Visible','on') ;
        set(handles.nG,'Visible','on') ;
    else
        handles.wu = 1 ;
        set(handles.wU,'Value',1) ;
        set(handles.text8,'Visible','off') ;
        handles.wg = 0 ;
        set(handles.wG,'Value',0) ;
        set(handles.wO,'Value',0) ;
        set(handles.wT,'Value',0) ;
        set(handles.wS,'Value',0) ;
        set(handles.wX,'Value',0) ;
        set(handles.wP,'Value',0) ;
        set(handles.nG,'Visible','off') ;
        set(handles.nO,'Visible','off') ;
        set(handles.nS,'Visible','off') ;
        set(handles.oS,'Visible','off') ;
        set(handles.nX,'Visible','off') ;
    end
end

% if offset, tail, or factor selected, turn on txt box if Gau also on
% Turn slope on only if exp tail selected
if strcmp(h,'wo') == 1 
    if num == 1
        if get(handles.wG,'Value') == 1
            set(handles.nO,'Visible','on') ;
            handles.osv = handles.os ;
        else
            set(handles.nO,'Visible','off') ;
            set(handles.oS,'Visible','off') ;
            set(handles.wO,'Value',0) ;
            %handlse.osv = 0 ;
        end
    end
end
if strcmp(h,'wt') == 1 
    if num == 1
        if get(handles.wG,'Value') == 1
            set(handles.nS,'Visible','on') ;
        else
            set(handles.nS,'Visible','off') ;
            set(handles.oS,'Visible','off') ;
            set(handles.wT,'Value',0) ;
            set(handles.wS,'Value',0) ;
        end
    else
        set(handles.nS,'Visible','off') ;
        set(handles.oS,'Visible','off') ;
        set(handles.wT,'Value',0) ;
        set(handles.wS,'Value',0) ;
    end
end
if strcmp(h,'wi') == 1 
    if num == 1
        if get(handles.wG,'Value') == 1
            set(handles.wX,'Value',0) ;
            set(handles.nX,'Visible','off') ;
            handles.wx = 0 ;
        else
            set(handles.wI,'Value',0) ;
            set(handles.wX,'Value',1) ;
            set(handles.nX,'Visible','on') ; 
        end
    end
end

if strcmp(h,'ws') == 1 
    if num == 1
        if (get(handles.wG,'Value') == 1 && ...
                get(handles.wO,'Value') == 1 && ...
                get(handles.wT,'Value') == 1)
            set(handles.wS,'Value',1) ;
            set(handles.oS,'Visible','on') ;
            set(handles.wX,'Value',0) ;
            set(handles.nX,'Visible','off') ;
            handles.wx = 0 ;
        else
            set(handles.wS,'Value',0) 
            set(handles.oS,'Visible','off') ;   
        end
    else
        set(handles.wS,'Value',0) 
        set(handles.oS,'Visible','off') ;   
    end
end

if strcmp(h,'wx') == 1 
    if num == 1
        if get(handles.wG,'Value') == 1
            set(handles.wI,'Value',0) ;
            set(handles.nX,'Visible','on') ;
        else
            num = 0 ;
            set(handles.wX,'Value',0) ;
            set(handles.nX,'Visible','off') ;
        end
    else
        set(handles.nX,'Visible','off') ;
        set(handles.wI,'Value',1) ;
    end
end

% 3D plot if Gaussian selected
if strcmp(h,'wp') == 1 
    if num == 1
        if get(handles.wG,'Value') == 0
            set(handles.wP,'Value',0) ;
        end
    end
end

% Fully sampled center
if strcmp(h,'wc') == 1
    if num == 1       
        set(handles.nCL,'Visible','on') ;
        set(handles.nCP,'Visible','on') ;
        set(handles.wSqu,'Visible','on') ;
        set(handles.wCir,'Visible','on') ;        
        if get(handles.wCir,'Value') == 1
            set(handles.nCP,'Visible','off') ;
        end
    else
        set(handles.nCL,'Visible','off') ;
        set(handles.nCP,'Visible','off') ;   
        set(handles.wSqu,'Visible','off') ;
        set(handles.wCir,'Visible','off') ;       
    end
end
if strcmp(h,'wsqu') == 1
    if num == 1
        set(handles.wCir,'Value',0) ;
        set(handles.nCP,'Visible','on') ;  
    else
       set(handles.wSqu,'Value',0) ;
       set(handles.wCir,'Value',1) ;
       set(handles.nCP,'Visible','off') ;   
    end
end
if strcmp(h,'wcir') == 1
    if num == 1
        set(handles.wSqu,'Value',0) ;
        set(handles.nCP,'Visible','off') ;
    else
       set(handles.wSqu,'Value',1) ;
       set(handles.nCP,'Visible','on') ;
    end
end

% 3d point density (Disabled for now)
if  strcmp(h,'pd_3d') == 1
    if num == 1
        set(handles.pD_3d,'Value',0) ;
        handles.pd_3d = 0 ;
    end
end

% Save options (Save one file at a time)
if  strcmp(h,'tkp') == 1
    if num == 1
        set(handles.tKP_name,'Visible','on') ;
        set(handles.tTP_name,'Visible','off') ;
        set(handles.tTP,'Value',0) ;   
        handles.ttp = 0 ;
    else
        set(handles.tKP_name,'Visible','off') ;
    end
end
if  strcmp(h,'ttp') == 1
    if num == 1
        set(handles.tTP_name,'Visible','on') ;
         set(handles.tKP_name,'Visible','off') ;
        set(handles.tKP,'Value',0) ;   
        handles.ktp = 0 ;
    else
        set(handles.tTP_name,'Visible','off') ;
    end
end
          
% Update handles structure.
han_str = ['handles.' h, '= num', ';'] ;
eval(han_str) ;
guidata(hObject, handles) ;
% ***********************************************************

% --- Executes on button press in plot.
function d = plot_Callback(hObject, eventdata, handles)
% hObject    handle to plot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(hObject,'BackgroundColor','white');

% Program expects partitions to be equal or greater than lines
if handles.nl > handles.np
    errordlg('Lines must not exceed Paritions', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end

% Retrieve handles explicitly (in case plot not colled from gui)
handles = guidata(hObject) ;

% Initialize values
ha = 0 ;
hb = 0 ;
a = 0 ;
b = 0 ;
%c = 0 ;
d = 0 ;

% Plot (Note that nl,np reduced by half in this routine)
nl = (handles.nl)/2 ;
np = handles.np/2 ;
ns = handles.ns ;
nx = handles.nx ;
%osv = handles.os ;
if get(handles.wS,'Value')== 1
    os = handles.os ;
else
    os = 0 ;
end

if handles.wu == 1
    y = ones(1,np) ;
else
    a = np^2/((10*handles.ng)^2) ;
    % See if offset desired
    if handles.wo == 1
        d = handles.no ;
    else
        d = 0 ;
    end 
    ha = 1-d ;
    % See if exp tail desired 
    if handles.wt == 1
        xs = round(ns*np) ;
        x = 1:xs ;
        ya = ha*exp(-x.^2/a) + d ;
        % Add exponential tail
        hb = ha*exp(-xs^2/a) ;   
        b = a*hb/(2*xs*ha*exp(-xs^2/a)) ; 
        z = 1:(np-xs) ;
        yb = hb*exp(-z./b) + d ;
        y = [ya yb] ;
        % Get yb value in k-space corner for later check 
        z_max = ceil(sqrt(nl^2 + np^2)) - xs ;
        yb_min = hb*exp(-z_max/b) + d ;
    else
    % No exp tail
    x = 1:np ;
    y = ha*exp(-x.^2/a) + d ;
    end
    % Add slope if called for
    if get(handles.wS,'Value') == 1
        x = 1:np ;
        y = y - os*x/np ;
        % Must check that y remains positive to k-space corner
        yb_min = yb_min - os*ceil(sqrt(nl^2 + np^2))/np ;
        if yb_min <= 0
            errordlg('Weighting must be positive', 'Bad Input')
            set(hObject, 'BackgroundColor', 'r') ;
            return
        end
    end
end

% Do plot
plot(handles.axes1,y) ;
title('Weighting') ;
    set(handles.axes1,'YLim',[0 1.1])
    set(handles.axes1,'XLim',[0 np]) ;    
hold on

% Calculate x-factor
if handles.wx == 0 && np ~= nl
    xi = 0:(np/nl):np-1 ;
    xl = interp1(y,xi) ;
    plot(handles.axes1,xl,'r') ;  
elseif handles.wx == 1
    % Plot only if np unequal to nl and nx unequal to 1
    if nx ~= 1
        y = y(1:round(nx*np)) ;
        xi = 0:(nx*np/nl):(nx*(np-1));
        xl = interp1(y,xi) ;
        plot(handles.axes1,xl,'r') ;    
    end
end
hold off

% Put values into handles structure
handles.ha = ha ;
handles.hb = hb ;
handles.a = a ;
handles.b = b ;
handles.d = d ;
% Update handles structure.
guidata(hObject, handles) ;

% See if 3D plot desired 
if get(handles.wP,'Value') == 1
    % Do 3D plots, with handles.patplot = 1
    handles.patplot = 1 ;    
    guidata(hObject, handles) ;
    % Call pattern_calc
    pattern_calc(hObject, eventdata, handles) ;
    handles.patplot = 0 ;
end

guidata(hObject, handles) ;

% END OF FUNCTION PLOT_CALLBACK
% **********************************************************


function kval = pattern_calc(hObject, eventdata, handles)
% Calculates pattern 
        
% Retrieve handles explicitly (in case plot called from Calculate)
handles = guidata(hObject) ;
       
% Local variables
np = (handles.np)/2 ;   nl = (handles.nl)/2 ;
ha = handles.ha ;   hb = handles.hb ;
os  = handles.os ;
a = handles.a ;
b = handles.b ;
d = handles.d ; 
ns = handles.ns ;
nx = handles.nx ;

krad = zeros(np,nl) ;
krad_l = -nl:nl-1 ; 
krad_p = -np:np-1 ;
if handles.wx == 0
    krad_l = krad_l*np/nl ;
else
    krad_l = krad_l*(nx*np/nl) ;
end
krad_lm = repmat(krad_l,2*np,1) ;
krad_pm = repmat(krad_p',1,2*nl) ;    
krad = sqrt(krad_lm.^2 + krad_pm.^2) ;

% Check that no exponential tail requested
if handles.wt == 0
    kval = ha*exp(-krad.^2/a) + d ;
else
    % Tail is to be included
    rad = ns*np ;
    rad_small = zeros(2*np,2*nl) ;
    rad_large = zeros(2*np,2*nl) ;
    ind_s = find(krad < rad) ;
    rad_small(ind_s) = krad(ind_s) ;
    ind_l = find(krad > rad) ;
    krad_red = krad - rad ;
    rad_large(ind_l) = krad_red(ind_l) ;
    rad_small(ind_s) = ha*exp(-rad_small(ind_s).^2/a) + d ;
    rad_large(ind_l) = hb*exp(-rad_large(ind_l)./b) + d ;
    % Add slope if called for
    if get(handles.wS, 'Value') == 1
        rad_small(ind_s) = rad_small(ind_s) - os/np*krad(ind_s) ;
        rad_large(ind_l)= rad_large(ind_l)- os/np*krad(ind_l) ;
    end
    kval = rad_small + rad_large ;
end

% % Put kval into handles structure
% handles.kval = kval ;
% guidata(hObject, handles) ;

% Plot if function called from PLOT
if handles.patplot == 1
    figure ;    surf(kval) ; axis([0 2*nl 0 2*np 0 1]) ;
    title('Pattern for Adjustment of Step Size') ;
end

% END OF PATTERN_CALC ROUTINE
% ************************************************

% ***************************************************
% Calculate Poisson disk distribution
% --- Executes on button press in calc.
function calc_Callback(hObject, eventdata, handles)
% hObject    handle to calc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i = sqrt(-1) ;
% set(hObject,'BackgroundColor','white');
set(hObject,'BackgroundColor','green');
pause(.001) ;

% Program expects partitions to be equal or greater than lines
if handles.nl > handles.np
    errordlg('Lines must not exceed Paritions', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end

% set(hObject,'BackgroundColor','green')for calculation duration ;

% Local variables.  Note that s_size is initial step (radius)
nl = handles.nl ;
np = handles.np ;

s_size = handles.nss ;
n_neighbors = handles.npp ;

% Always call Plot
d = plot_Callback(hObject, eventdata, handles) ;

% Call pattern_calc if pattern is not uniform
if get(handles.wG,'Value') == 1
    kval = pattern_calc(hObject, eventdata, handles) ;
else 
    kval = ones(np,nl) ;
end

% Initialize arrays (pop_matrix (0 or1), pt_matrix (x + i*y))
pop_m = zeros(np,nl) ;
pt_m = zeros(np,nl) ;

% First point is at center of k-space (This is also first initiating pt)
pop_m(np/2+1,nl/2+1) = 1 ;
pt_m(np/2+1,nl/2+1) = 0 ;

% Random numbers for radius (from s_size to 2*s_size)(rad) and angle (ang)
rad = s_size + s_size*rand(1,n_neighbors) ;
ang = 2*pi*rand(1,n_neighbors) ;

% Next point kept as long as it is not in center cell (0,0)
k = 1 ;
px = rad(1)*cos(ang(1)) ; py = rad(1)*sin(ang(1)) ;
while (round(px)< 1 && round(py) < 1)
    k = k+1 ;
    if k > n_neighbors-1
        errordlg('Radius too small') ;
        return
    else
    px = rad(k)*cos(ang(k)) ; py = rad(k)*sin(ang(k)) ;
    end
end
pop_m(np/2+1+round(py),nl/2+1+round(px)) = 1 ;
pt_m(np/2+1+round(py),nl/2+1+round(px)) = px+i*py ;
ip = px + i*py ;

% New points kept if they are far enough away from other points
% Step size set from position of initiating point
for j = k+1:n_neighbors
    px = rad(j)*cos(ang(j)) ; py = rad(j)*sin(ang(j)) ;
    pp = px+i*py ;
    m = np/2+1+round(py) ;  n = nl/2+1+round(px) ;
    ss = s_size ;
    % Neighborhood points are within (2*round(ss)+1)^2 points
    rc = round(ss) ;
    % Call funct add_pt; gives pflag = 1 for good points
    pflag = add_pt(nl, np, ss, rc, pp, pop_m, pt_m, ip(1)) ;
    if pflag == 1
        % Good point; add to po_m, pt_m, and ip (initiating point list)
        pop_m(np/2+1+round(imag(pp)),nl/2+1+round(real(pp))) = 1 ;
        pt_m(np/2+1+round(imag(pp)),nl/2+1+round(real(pp))) = pp ;
        ip = [ip pp] ;
    end
end

% SUBSEQUENT POINTS ***************************************
while ~isempty(ip)
   % length(ip)
    % Initiating point from ip; remove point from ip list
    pxi = real(ip(1)) ; pyi = imag(ip(1)) ;
    ipp = ip(1);    ip(1) = [] ;
    ss = s_size/kval(np/2+1+round(pyi),nl/2+1+round(pxi)) ;
    rad = ss + ss*rand(1,n_neighbors) ;
    ang = 2*pi*rand(1,n_neighbors) ;  
    for j = 1:n_neighbors
        px = rad(j)*cos(ang(j)) ; py = rad(j)*sin(ang(j)) ;
        pp = px+pxi+i*(py+pyi) ;
        % Get proposed point position and rc
        m = np/2+1+round(imag(pp)) ;  n = nl/2+1+round(real(pp)) ;
        rc = round(ss) ; 
        % Call funct add_pt
        pflag = add_pt(nl, np, ss, rc, pp, pop_m, pt_m, ipp) ;
        if pflag == 1
            % Good point; add to po_m, pt_m, and ip
            pop_m(np/2+1+round(imag(pp)),nl/2+1+round(real(pp))) = 1 ;
            pt_m(np/2+1+round(imag(pp)),nl/2+1+round(real(pp))) = pp ;            
            ip = [ip pp] ;   
        end
    end
end

% Check for desired outputs ************************
% First see if fully sampled center desired
if handles.wc == 1 
    ncp = (handles.ncp)/2 ;     ncl = (handles.ncl)/2 ;
    if get(handles.wCir,'Value') == 0
        ncp = (handles.ncp)/2 ;     ncl = (handles.ncl)/2 ;
        pop_m(np/2+1-ncp:np/2+ncp,nl/2+1-ncl:nl/2+ncl) = 1 ;
        % Have to do reals and imaginaries separately
        pt_m(np/2+1-ncp:np/2+ncp,nl/2+1-ncl:nl/2+ncl) = ...
            i*repmat((-ncp:ncp-1)',1,2*ncl) ...
            + repmat((-ncl:ncl-1),2*ncp,1) ;
    else
        % Create full k-space matrix to get center points
        l = -nl/2:nl/2-1 ;    p = -np/2:np/2-1 ;
        lm = repmat(l,np,1) ;
        pm = repmat(p',1,nl) ;
        kspace = i*pm + lm ;
        size(kspace)
        kind = find(abs(kspace)<= ncl) ;
        pop_m(kind) = 1 ;
        pt_m(kind) = kspace(kind) ;
    end
end

if handles.tp == 1
    set(handles.tPD, 'String',num2str(nnz(pop_m))) ;
end
if handles.ta == 1
    accel = handles.nl*handles.np/nnz(pop_m) ;
    set(handles.tAD, 'String',num2str(accel)) ;
end
if handles.tk == 1
    figure ;    spy(pop_m) ;    title('K-Space Points') ;
end

% 1D point density 
if get(handles.pD_1d,'Value') == 1
    no_l = round(nl/10) ;
    no_p = round(np/10) ;
    % Partitions (y) direction
    pd_l1 = pop_m(np/2+1:np-1,nl/2-no_l:nl/2+no_l) ;
    
    pd_l2 = flipud(pop_m(3:np/2+1,nl/2-no_l:nl/2+no_l)) ;
  
    pd_l = (pd_l1+pd_l2)' ;
    pd_l = sum(pd_l)/(2*(2*no_l+1)) ;
    figure ;    plot(pd_l) ; title('Approximate Point Density') ;
    hold on
    % Lines (x) direction
    pd_p1 = pop_m(np/2-no_p:np/2+no_p,nl/2+1:nl-1) ;
   
    pd_p2 = fliplr(pop_m(np/2-no_p:np/2+no_p,3:nl/2+1)) ;
   
    pd_p = pd_p1+pd_p2 ;
    pd_p = sum(pd_p)/(2*(2*no_p+1)) ;
    plot(pd_p, 'r') ;   
    axis([0 np/2 0 1]) ;   
end    

% Make pt_m matrix (x+i*y)into integers
pt_m = round(pt_m) ;

% Put pop_m and pt_m into handles structure
%OOPS - INCORRECT K-SPACE CONVENTION
handles.pop_m = pop_m ;
handles.pt_m = pt_m ;

% Update handles structure
guidata(hObject, handles);


% 3D point density ((to be done))

set(hObject,'BackgroundColor','white');

% END OF CALC FUNCTION
% ***********************************************************

% ***************************************************************
% Function add_pt
function pflag = add_pt(nl, np, rm, rc, pp, pop_m, pt_m, ip)
% To add a point if other neighborhood points far enough away

% Neighborhood is (2*rc+1)^2 cells, where rc is an integer
% Possible point position is pp (x + i*y) (k-space)
pflag = 1 ;

% Get cell (m,n) of possible point
px = real(pp) ;     py = imag(pp) ;
m = np/2 + 1 + round(py) ;    n = nl/2 + 1 + round(px) ;

% Ensure that possible point is not in same cell as initiating point
if rm < sqrt(2)
    if (np/2+1+round(imag(ip)) == m && nl/2+1+round(real(ip)) == n)
        pflag = 0 ;
        return
    end
end

% See if proposed point is within FOV
if (m < 1 || m > np || n < 1 || n > nl)
    pflag = 0 ;
    return
end

% Look over neighborhood for points, calculate radius and set pflag.
for r = (m-rc):(m+rc)
    for s = (n-rc):(n+rc)
        % Make sure cell is within FOV
        if (r > 0 && r < np+1 && s > 0 && s < nl+1)  
            if pop_m(r,s) == 1
                pxn = real(pt_m(r,s));  pyn = imag(pt_m(r,s)) ;
                rad = sqrt((pxn-px)^2 + (pyn-py)^2) ;
                % If too close, reject
                if rad < rm 
                    pflag = 0 ;
                    return
                end
            end
        end
    end
end
        
% END OF FUNCTION
% ********************************************************************


% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(hObject, 'BackgroundColor', 'w') ;

% Check to see if something to save selected
if handles.tkp == 0 && handles.ttp == 0
    errordlg('Nothing selected to save', 'Bad Input')
    set(hObject, 'BackgroundColor', 'r') ;
    return
end

% ((From mr-parameters))
 mpathname = mfilename('fullpath') ;
 mpathname = mpathname(1:length(mpathname)-12) ;
 mmpathname = [mpathname '\Distributions'] ;
 cd(mmpathname) ;
 [filename, pathname] = uiputfile(get(handles.tKP_name,'String'), 'Select a File') ;  
if filename == 0    % Cancel was clicked
    cd(mpathname) ;
    return
else
    % Check that the correct subdirectory is used
    plength = length(pathname) ;
    cdir = pathname(plength-27:plength) ;
    if strcmp('\Poisson_Disk\Distributions\', cdir)== 0
         errordlg('Unexpected File Location', 'Bad Input File') ;
         cd(mpathname) ;
        return
    else 
        if handles.tkp == 1
            kspace_matrix = handles.pt_m ;
            save(filename, 'kspace_matrix') ;
        end
        if handles.ttp == 1
            population_matrix = handles.pop_m ;
            save(filename, 'population_matrix') ;
        end      
    end ;
    cd(mpathname) ;
end ;


% Save as k-space points
function tKP_name_Callback(hObject, eventdata, handles)
% hObject    handle to tKP_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tTP_name as text
%        str2double(get(hObject,'String')) returns contents of tKP_name as a double

% --- Executes during object creation, after setting all properties.
function tKP_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tKP_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tTP_name_Callback(hObject, eventdata, handles)
% hObject    handle to tTP_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tTP_name as text
%        str2double(get(hObject,'String')) returns contents of tTP_name as a double

% Save points matrix (ones and zeros)
% --- Executes during object creation, after setting all properties.
function tTP_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tTP_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pD_3d.
function pD_3d_Callback(hObject, eventdata, handles)
% hObject    handle to pD_3d (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of pD_3d


% --- Executes during object creation, after setting all properties.
function oS_CreateFcn(hObject, eventdata, handles)
% hObject    handle to oS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
