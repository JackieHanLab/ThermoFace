function [vis_scale] = read_data(filename,savegraypath)
%function [vis, ir, meta1, meta2] = read_is2(filename)
%filename   - (FLUKE Ti10 .IS2-file)
%vis        - visual spectrum image (Grayscale 640x450 16-bit)
%ir         - themal ir image (Grayscale 160x120 16-bit)
%meta1      - metadata at start of file 
%meta2      - metadata before IR-data
%
%Example:
% [vis] = read_data('D:\Thermal_project\is2test\Images\Main\IR.data');
% figure(1);imagesc(vis);colormap gray;figure(2);imagesc(ir);colormap hot;


f = fopen(filename);
d = uint8(fread(f,'uint8'));
fclose(f);

tename = replace(filename,'IR.data','IRImageInfo.gpbenc');
f = fopen(tename);
tedata = uint8(fread(f,'uint8'));
fclose(f);


meta1_start = 1;
meta1_len = 1280;
meta1_end = meta1_start+meta1_len-1;

vis_start = meta1_end+1;
vis_dim = [640 480];
vis_datatype = 'uint16';
vis_datatype_size = 2;
vis_end = vis_start+prod(vis_dim)*vis_datatype_size-1;
ix = vis_start:vis_end;
vis = typecast(d(ix),vis_datatype);
vis = reshape(vis,vis_dim(1), vis_dim(2))';
% turn into Tc
fcal = './readis2_ti401pro.dat';
tcal = load(fcal);
tc = interp1(tcal(:,1),tcal(:,2),double(vis),'linear','extrap');

% get the background temperature
% teall =  typecast(tedata,vis_datatype);
te=  16816;%teall(21);
% te to Te
Te=0.12556*double(te)-2089.4136 ;



%tb=im2double(tb);

% emissivity 0.97, transmittance 100%, ambient temperature change
%Tb=Tc/(e*t)-(2-e-t)*Te
tb=tc/0.97-0.03*Te;



% Normalized to 27-38 degree Celsius range

% above 27 degrees Celsius be recorded
% Amin = 0.0717;%4702; % 27 Celsius
% Amax = 0.0850;%5591; % 38 Celsius
vis_scale = (tb - 27)/11;
vis_scale = repmat(vis_scale,[1,1,3]);

% get file name
b=regexpi(filename,'.*?(?=/Images)','match');
indfir=max(strfind(b{1,1},'/'));
smap_name=b{1,1}(indfir+1:length(b{1,1}));

imwrite(vis_scale,strcat(savegraypath,smap_name,'.png'));
return 





