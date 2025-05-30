; setup.iss - Inno Setup script for G.A.I.A Windows installer
; This script creates a Windows installer for the G.A.I.A application, including Python environment,
; source code, browser extension, and assets.

[Setup]
AppName=G.A.I.A
AppVersion=2.4.0
AppPublisher=Your Name
AppPublisherURL=https://github.com/yourusername/gaia
AppSupportURL=https://github.com/yourusername/gaia/issues
AppUpdatesURL=https://github.com/yourusername/gaia/releases
DefaultDirName={autopf}\G.A.I.A
DefaultGroupName=G.A.I.A
OutputDir=dist
OutputBaseFilename=GAIA_Setup_v2.4.0
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=assets\icons\icon128.ico
UninstallDisplayIcon={app}\icon128.ico
LicenseFile=LICENSE
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Core application files
Source: "src\core\gaia.py"; DestDir: "{app}\src\core"; Flags: ignoreversion
Source: "src\core\__init__.py"; DestDir: "{app}\src\core"; Flags: ignoreversion
Source: "src\utils\helpers.py"; DestDir: "{app}\src\utils"; Flags: ignoreversion
Source: "src\utils\__init__.py"; DestDir: "{app}\src\utils"; Flags: ignoreversion

; Browser extension files
Source: "src\extension\manifest.json"; DestDir: "{app}\src\extension"; Flags: ignoreversion
Source: "src\extension\background.js"; DestDir: "{app}\src\extension"; Flags: ignoreversion
Source: "src\extension\content.js"; DestDir: "{app}\src\extension"; Flags: ignoreversion
Source: "src\extension\popup\popup.html"; DestDir: "{app}\src\extension\popup"; Flags: ignoreversion
Source: "src\extension\popup\popup.js"; DestDir: "{app}\src\extension\popup"; Flags: ignoreversion

; Assets
Source: "assets\gaia_logo.png"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "assets\icons\icon16.png"; DestDir: "{app}\assets\icons"; Flags: ignoreversion
Source: "assets\icons\icon48.png"; DestDir: "{app}\assets\icons"; Flags: ignoreversion
Source: "assets\icons\icon128.png"; DestDir: "{app}\assets\icons"; Flags: ignoreversion
Source: "assets\icons\icon128.ico"; DestDir: "{app}"; Flags: ignoreversion

; Dependencies and configuration
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion

; Virtual environment (pre-built, if included)
Source: "venv\*"; DestDir: "{app}\venv"; Flags: ignoreversion recursesudirs createallsubdirs

; Log and version directories (empty, with .gitkeep)
Source: "logs\.gitkeep"; DestDir: "{app}\logs"; Flags: ignoreversion
Source: "versions\.gitkeep"; DestDir: "{app}\versions"; Flags: ignoreversion

[Icons]
Name: "{group}\G.A.I.A"; Filename: "{app}\venv\Scripts\python.exe"; Parameters: """{app}\src\core\gaia.py"""; WorkingDir: "{app}"; IconFilename: "{app}\icon128.ico"
Name: "{group}\{cm:UninstallProgram,G.A.I.A}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\G.A.I.A"; Filename: "{app}\venv\Scripts\python.exe"; Parameters: """{app}\src\core\gaia.py"""; WorkingDir: "{app}"; IconFilename: "{app}\icon128.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\venv\Scripts\python.exe"; Parameters: "-m pip install -r ""{app}\requirements.txt"""; Description: "Install Python dependencies"; Flags: runhidden nowait postinstall
Filename: "{app}\venv\Scripts\python.exe"; Parameters: """{app}\src\core\gaia.py"""; Description: "{cm:LaunchProgram,G.A.I.A}"; Flags: nowait postinstall skipifsilent

[Dirs]
Name: "{app}\logs"
Name: "{app}\versions"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
  begin
    // Ensure Python is available in venv
    if not FileExists(ExpandConstant('{app}\venv\Scripts\python.exe')) then
    begin
      MsgBox('Python virtual environment not found. Please ensure venv is included or install Python dependencies manually.', mbError, MB_OK);
    end;
  end;
end;
