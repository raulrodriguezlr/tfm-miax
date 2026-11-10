% Plantilla para la memoria del TFM MIAX de BME
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tfm}[2026/06/29 Plantilla para la memoria del TFM MIAX]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass{book}

\newcommand{\@tutorlabel}{\ClassError{tfm}{Falta especificar
\string\tutorlabel{}}
 {Usa el comando \string\tutorlabel{} en el preambulo para especificar
\MessageBreak la etiqueta asociada (Tutor, Tutora, Tutores)}}
\newcommand{\tutorlabel}[1]{\gdef\@tutorlabel{#1}}

\newcommand{\@authorlabel}{\ClassError{tfm}{Falta especificar
\string\authorlabel{}} 
 {Usa el comando \string\authorlabel{} en el preambulo para especificar
\MessageBreak la etiqueta asociada (Autor, Autora)}}
\newcommand{\authorlabel}[1]{\gdef\@authorlabel{#1}}

\newcommand{\@tutor}{\ClassError{tfm}{Falta especificar
\string\tutor{}}
 {Usa el comando \string\tutor{} en el preambulo para especificar
\MessageBreak el nombre del tutor o la tutora.}}
\newcommand{\tutor}[1]{\gdef\@tutor{#1}}

\newcommand{\@convocatoria}{\ClassError{tfm}{Falta especificar
\string\convocatoria{}}
 {Usa el comando \string\convocatoria{} en el preambulo para especificar
\MessageBreak el mes y año de la lectura.}}
\newcommand{\convocatoria}[1]{\gdef\@convocatoria{#1}}


\newcommand{\@titulacion}{\ClassError{tfm}{Falta especificar
\string\titulacion{}}
 {Usa el comando \string\titulacion{} en el preambulo para especificar
\MessageBreak la titulacion.}}
\newcommand{\titulacion}[1]{\gdef\@titulacion{#1}}


\RequirePackage[T1]{fontenc}
\RequirePackage[spanish,activeacute,es-tabla]{babel}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{fancyhdr}
\RequirePackage{listings}
\RequirePackage{hyperref}
\RequirePackage[pdftex]{graphicx}
\RequirePackage{tikz}
\RequirePackage{framed}
\RequirePackage{geometry}
\RequirePackage{soul}
\RequirePackage{bold-extra}
\RequirePackage{ragged2e}
\RequirePackage{longtable}


\usepackage[utf8]{inputenc} %Codificacion de entrada

% Ruta donde buscar las figuras
\graphicspath{{figs/}}
\DeclareGraphicsExtensions{.pdf,.jpg,.png}


 \geometry{
  a4paper,
  total={160mm,255mm},
  left=25mm,
  right=25mm,
  top=25mm,
  bottom=20mm
 }

\newlength{\ancho}
\setlength{\ancho}{16cm}


\hypersetup{bookmarks=true,%
bookmarksnumbered=true,%
colorlinks,%
citecolor=red,%
filecolor=blue,%
linkcolor=blue,%
urlcolor=blue,%
pdfstartview=FitH,%
pdffitwindow=true,%
pdfauthor=\@author}

\setlength{\parskip}{5pt}



% ============================================================================
% CONFIGURACIÓN DE COLORES (Adaptable a los colores corporativos de BME)
% ============================================================================
\definecolor{bme-blue}{RGB}{56, 90, 140} 
\definecolor{bme-dark}{RGB}{30, 60, 90}
\definecolor{bme-light}{RGB}{220, 230, 240}
\definecolor{accent}{RGB}{190, 80, 70}

% ============================================================================
% DEFINICIÓN DE COMANDOS PERSONALIZADOS
% ============================================================================
\newcommand{\titulo}[1]{\textbf{\Large\color{bme-dark}#1}}
\newcommand{\subtitulo}[1]{\textbf{\small\color{bme-blue}#1}}
\newcommand{\seccion}[1]{\large\textbf{\color{bme-dark}#1}}
\newcommand{\campo}[1]{\underline{#1}}


\newcommand{\portada}{
  \pagestyle{empty}
   \makeatletter
    \centering 

   \begin{left}
     % Logo principal de la institución (Instituto BME / BME)
     \includegraphics[width=0.25\ancho]{figs/Logotipo-Instituto-BME-300x300.jpg}
   \end{left}



   \vspace*{1\baselineskip} 

   \begin{minipage}[h]{\ancho}
     \centering
     \LARGE
     \capsdef{T1/ppl/m/n/5-15}{\scshape}{.16em}{.6em}{.2em}
     \scshape \@titulacion
   \end{minipage}

   \vspace*{2\baselineskip} 



   \vspace*{3\baselineskip} 

    {\scshape \LARGE Trabajo Fin de Máster}

   \vspace*{2\baselineskip} 

    % ------------------------------------------------
    %   Title
    %------------------------------------------------
    \rule{\ancho}{1.6pt}\vspace*{-\baselineskip}\vspace*{2pt}
    \rule{\ancho}{0.4pt} 

    \vspace{0.75\baselineskip} 

   \begin{minipage}[h]{\ancho}
     \centering
     \LARGE
     \capsdef{T1/ppl/m/n/5-15}{\scshape}{.16em}{.6em}{.2em}
     \scshape \@title
   \end{minipage}

    \vspace{0.75\baselineskip} 

    \rule{\ancho}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt} 
    \rule{\ancho}{1.6pt} 

   \vfill

   \hspace*{0.15\ancho}\begin{minipage}[h]{0.85\ancho}
   \large
    \bfseries{\hspace{5cm}\scshape \@authorlabel:\\ 
    \mbox{}\hspace{5cm}\@author}

    \vspace{0.5\baselineskip} 

    {\hspace{5cm}\scshape \@tutorlabel:\\ 
    \mbox{}\hspace{5cm}\@tutor}

    % \vspace{0.5\baselineskip} 
    % {\scshape \@convocatoria}
 \end{minipage}
 \makeatother
 }


 \newcommand{\contraportada}{
   \pagestyle{empty}
   \makeatletter
    \centering 
    \vspace*{0.2\baselineskip} 

   \begin{center}
     \includegraphics[width=0.5\ancho]{figs/Logotipo-Instituto-BME-300x300.jpg}
   \end{center}

   \vspace*{\baselineskip} 

   \titulacion

   \vspace*{2\baselineskip} 

    {\scshape \LARGE Trabajo Fin de Máster}

   \vspace*{2\baselineskip} 

    % ------------------------------------------------
    %   Title
    %------------------------------------------------
    \rule{\ancho}{1.6pt}\vspace*{-\baselineskip}\vspace*{2pt} 
    \rule{\ancho}{0.4pt} 

    \vspace{2\baselineskip} 

   \begin{minipage}[h]{\ancho}
     \centering
     \LARGE
     \capsdef{T1/ppl/m/n/5-15}{\scshape}{.16em}{.6em}{.2em}
     \scshape \@title
   \end{minipage}

    \vspace{0.5\baselineskip} 

    \rule{\ancho}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt} 
    \rule{\ancho}{1.6pt} 

    \vspace{5\baselineskip} 

   \hspace*{0.15\ancho}\begin{minipage}[h]{0.85\ancho}
   \large
    \bfseries{\scshape \@authorlabel: \@author} 

    \vspace{0.5\baselineskip} 

    {\scshape \@tutorlabel: \@tutor} 

    \vspace{0.5\baselineskip} 
 \end{minipage}

    \vfill

    \rule{\ancho}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt}

    \vspace{2\baselineskip} 

   \vfill
   \makeatother
   \mdseries
 }

\newcommand{\ia}{\input{tex/declaracion_ia.tex}}

 \newenvironment{resumen}
    {\begin{center}\begin{tabular}{p{0.9\textwidth}}
       \hline\\
       {\bf Resumen:} \\[3mm]
    }
    {
    \\\\\hline
    \end{tabular}
  \end{center}
    }

 \newenvironment{abstract}
    {\begin{center}\begin{tabular}{p{0.9\textwidth}}
       \hline\\
       {\bf Abstract:} \\[3mm]
    }
    {
    \\\\\hline
    \end{tabular}
  \end{center}
    }

  \newenvironment{agradecimientos}
    {\begin{center}\begin{tabular}{p{0.9\textwidth}}
       \hline\\
       {\bf Agradecimientos:} \\[3mm]
    }
    {
    \\\\\hline
    \end{tabular}
  \end{center}
    }

\lstdefinestyle{nonumbers} {numbers=none}
\lstset{basicstyle=\scriptsize\ttfamily}
\renewcommand{\UrlFont}{\scriptsize}

% Formato de los listados de código
\lstset{
  basicstyle=\ttfamily\small,
  columns=flexible,
  keywordstyle=\color{blue}\bf,
  commentstyle=\color{gray},
  stringstyle=\color{green},
  breaklines=true,inputencoding=utf8,
  frame=single,showstringspaces=false,
  style=nonumbers,extendedchars=true,
  literate={á}{{\'a}}1 {é}{{\'e}}1 {í}{{\'i}}1 {ó}{{\'o}}1 {ú}{{\'u}}1 {Á}{{\'A}}1 {É}{{\'E}}1 {Í}{{\'I}}1 {Ó}{{\'O}}1 {Ú}{{\'U}}1 {ñ}{{\~n}}1 {Ñ}{{\~N}}1,
}

\definecolor{enlaces}{rgb}{0.6,0.6,0.9}
\definecolor{colurl}{rgb}{0.34,0.36,0.60}

\hypersetup{
    colorlinks = true,
    linkcolor = enlaces,
    citecolor = enlaces,
    filecolor = enlaces,
    urlcolor = enlaces
}

\renewcommand{\lstlistingname}{Listado}
\renewcommand{\lstlistlistingname}{Índice de listados de código}

\lstloadlanguages{html,java,sql,c,xml,tex,matlab,python} % Añadido Python por si acaso
\lstdefinestyle{numbers} {numbers=left, stepnumber=1, numberstyle=\tiny, numbersep=10pt}
\lstdefinestyle{nonumbers} {numbers=none}

%% Para acentos en listings con UTF8
\lstset{
    inputencoding=utf8,
    extendedchars=true,
    literate={á}{{\'a}}1 {é}{{\'e}}1 {í}{{\'i}}1 {ó}{{\'o}}1 {ú}{{\'u}}1 {Á}{{\'A}}1 {É}{{\'E}}1 {Í}{{\'I}}1 {Ó}{{\'O}}1 {Ú}{{\'U}}1 {ñ}{{\~n}}1 {Ñ}{{\~N}}1
}

%%%%%%%   DEFINICION DE UN CODIGO SIN KEYWORDS
\lstdefinelanguage{plain}
   {morekeywords={},
   sensitive=false,
}

\pagestyle{fancy}

\fancypagestyle{tfm}{%
  %Información del capítulo en la parte exterior
  \fancyhead[EL,OR]{Capítulo \thechapter}
  % Información del número de página en la parte interior
  \fancyhead[OL,ER]{Página \thepage}
  \renewcommand{\headrulewidth}{1pt}
  \fancyfoot{}
}

\fancypagestyle{appendix}{%
  %Información del capítulo en la parte exterior
  \fancyhead[EL,OR]{Apéndices}
  % Información del número de página en la parte interior
  \fancyhead[OL,ER]{Página \thepage}
  \renewcommand{\headrulewidth}{1pt}
  \fancyfoot{}
}

\fancypagestyle{plain}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
}