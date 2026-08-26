; Inno Setup script for LabelVision Studio.
;
; Built by make_installer.bat, which passes AppVersion and Edition via /D.
; Do not run directly unless you define them yourself:
;   iscc installer\LabelVisionStudio.iss /DAppVersion=0.9.69 /DEdition=full

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Edition
  #define Edition "full"
#endif

#define AppName "LabelVision Studio"
#define AppExeName "LabelVisionStudio.exe"
#define AppPublisher "LabelVision"

[Setup]
AppId={{8E4C1D2A-9B7F-4A6E-B3D5-1C0F7A2E9B84}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=.
OutputBaseFilename=LabelVisionStudio-{#AppVersion}-{#Edition}-setup
SetupIconFile=..\label_detections\ui\assets\app.ico
UninstallDisplayIcon={app}\{#AppExeName}
WizardStyle=modern
; The full edition is ~3 GB of mostly-compressible tensor libraries. lzma2/max
; roughly halves it; solid compression helps further across the many small
; torch DLLs. This makes the installer build slow -- that is the tradeoff.
Compression=lzma2/max
SolidCompression=yes
; 64-bit only: PySide6/torch/pypylon ship no 32-bit wheels.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Per-machine install needs admin; that is what puts it under Program Files.
PrivilegesRequired=admin
DisableProgramGroupPage=yes
LicenseFile=
; Refuse to downgrade over a newer install.
VersionInfoVersion={#AppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; The entire PyInstaller output tree.
Source: "..\dist\LabelVisionStudio\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove PyInstaller's extraction leftovers, but never the user's data --
; captures/labels/exports live in %LOCALAPPDATA%\LabelVisionStudio and are
; deliberately left behind so an uninstall/reinstall cycle cannot destroy work.
Type: filesandordirs; Name: "{app}\_internal"

[Messages]
; Point users at where their work actually lives.
FinishedLabel=Setup has installed {#AppName} on your computer.%n%nYour captures, labels and exports are stored in:%n%%LOCALAPPDATA%%\LabelVisionStudio\data%n%nThis folder is kept when you uninstall or upgrade.

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
