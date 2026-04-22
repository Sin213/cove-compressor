; Inno Setup script for Cove Compressor (Windows)
; Invoked from build.ps1 via:
;   iscc /DAppVersion=X.Y.Z /DSourceDir=<abs dist\cove-compressor> \
;        /DOutputDir=<abs release> /DIconFile=<abs cove_icon.ico> installer.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\cove-compressor"
#endif
#ifndef OutputDir
  #define OutputDir "..\release"
#endif
#ifndef IconFile
  #define IconFile "..\cove_icon.ico"
#endif

[Setup]
AppId={{A71D4B02-3C98-4E6F-9B13-2F52A9C7D118}
AppName=Cove Compressor
AppVersion={#AppVersion}
AppPublisher=Cove
AppPublisherURL=https://github.com/Sin213/cove-compressor
AppSupportURL=https://github.com/Sin213/cove-compressor/issues
AppUpdatesURL=https://github.com/Sin213/cove-compressor/releases
DefaultDirName={autopf}\Cove Compressor
DefaultGroupName=Cove Compressor
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\cove-compressor.exe
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=cove-compressor-{#AppVersion}-Setup
SetupIconFile={#IconFile}
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Cove Compressor"; Filename: "{app}\cove-compressor.exe"
Name: "{group}\Uninstall Cove Compressor"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Cove Compressor"; Filename: "{app}\cove-compressor.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\cove-compressor.exe"; Description: "Launch Cove Compressor"; Flags: nowait postinstall skipifsilent
