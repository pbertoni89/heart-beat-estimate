%% Bug Conosciuti:
%%
%% 1- La grandezza delle intestazioni può essere troppo grande per titoli dei capitoli lunghi
%% 2- Il frontespizio fa uso di grandezze del font relative, in caso venga aumentata la grandezza
%%    del font, il frontespizio potrebbe non entrare in una singola pagina, va quindi regolato di conseguenza.
%% 3- In caso di titolo della tesi lungo e su più righe, è il caso di cambiare le grandezze dei font sul
%%    frontespizio per far si che le varie parti siano più spaziate

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesPackage{TesiDiaUniroma3}[2008/02/13 v1.0 Tesi - DIA-Uniroma3]

\LoadClass[a4paper,11pt,oneside,italian]{memoir}

\RequirePackage[italian]{babel}
\RequirePackage{graphicx}
\RequirePackage{setspace}

\usepackage[T1]{fontenc}
\usepackage[plainpages=false, hidelinks]{hyperref}
\usepackage{amsthm}	% per definizioni e teoremi
\usepackage{amsmath}	% per ``cases'' environment
\usepackage{amsfonts}  % per simbolo insieme reali
\usepackage{amssymb}	% math
\usepackage{xparse}    % intanto per varargs
\usepackage{float}     % per usare H nel figure env
%\usepackage{latexsym}
\usepackage{graphicx}
\usepackage{caption}	 % per figure in tabelle, math env in captions
%\usepackage{subcaption}%    "  "	
\usepackage{epstopdf}
%\usepackage[all]{hypcap}
\usepackage[utf8]{inputenc} % in teoria per usare accenti
\usepackage[T1]{fontenc}    %    "  "  "
\usepackage{patlib}		% :D

\linespread{1.4}
\setlrmarginsandblock{3cm}{3cm}{*}
% profondità dell'indice (settato per stampare fino alle sottosezioni)
\setcounter{tocdepth}{1}
% numera parti, capitoli, sezioni, sottosezioni e sotto-sottosezioni
\maxsecnumdepth{subsubsection}
\setsecnumdepth{subsubsection}

% header personalizzati
\let\footruleskip\relax % for compatibility of memoir and fancyhdr
\let\rm\rmfamily        % for compatibility of memoir and blindtext (demo only)
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\addtolength{\headheight}{14pt}
\fancyhead[R]{\textbf{\thepage}}
\fancyhead[L]{\textbf{\leftmark}}

\newcommand{\@annoAccademico}{2006/2007}
\newcommand{\annoAccademico}[1]{\renewcommand{\@annoAccademico}{#1}}

\newcommand{\@autore}{Cognome Nome}
\newcommand{\autore}[1]{\renewcommand{\@autore}{#1}}

\newcommand{\@titolo}{Titolo della tesi}
\newcommand{\titolo}[1]{\renewcommand{\@titolo}{#1}}

\newcommand{\@matricola}{000000}
\newcommand{\matricola}[1]{\renewcommand{\@matricola}{#1}}

\newcommand{\@relatore}{Prof. Nome Cognome}
\newcommand{\relatore}[1]{\renewcommand{\@relatore}{#1}}

\newcommand{\@dedica}{}
\newcommand{\dedica}[1]{\renewcommand{\@dedica}{#1}}

\newcommand{\generaFrontespizio}{
   \thispagestyle{empty}
   \begin{center}
      \vspace{10mm}
      {\includegraphics[width=0.3 \linewidth]{figure/logo_unibs.png}} \\
      \vspace{10mm}
      {\Large UNIVERSIT\`A DEGLI STUDI DI BRESCIA} \\
      \vspace{5mm}
      {\LARGE Facolt\`a di Ingegneria} \\
      {\Large Corso di Laurea in Ingegneria Informatica} \\
      \vspace{15mm}
      {\LARGE Tesi di Laurea Triennale} \\
      \vspace{15mm}
      {\HUGE \@titolo} \\
      \vspace{15mm}
      {\large Laureando \\ \textbf{\@autore} \\ Matricola \@matricola} \\
      \vspace{7mm}
      {\large Relatore \\ \textbf{\@relatore}} \\
      \vfill
      {\large Anno Accademico \@annoAccademico} \\
   \end{center}
   \newpage 
}

\newcommand{\generaDedica}{
   \thispagestyle{empty}
   \null\vspace{\stretch{1}}
   \begin{flushright}
      \emph{{\@dedica}}
   \end{flushright}
   \vspace{\stretch{2}}\null
   \newpage
}

\newcommand{\ringraziamenti}[1]{\newpage \chapter*{Ringraziamenti} \input{#1}}

\newcommand{\conclusioni}[1]{\newpage \chapter{Conclusioni e sviluppi futuri} \markboth{CONCLUSIONI E SVILUPPI FUTURI}{CONCLUSIONI E SVILUPPI FUTURI} \input{#1}}

\newcommand{\capitolo}[2]{\newpage \chapter{#1}\label{cap:#2}\input{#2}}

\newcommand{\appendiceA}[2]
	{\newpage \chapter{Appendice A \\ #1}  
	 \markboth{APPENDICE A}{APPENDICE A} \input{#2}}  
\newcommand{\appendiceB}[2]
	{\newpage \chapter{Appendice B \\ #1}  
	 \markboth{APPENDICE B}{APPENDICE B} \input{#2}}  
\newcommand{\appendiceC}[2]
	{\newpage \chapter{Appendice C \\ #1}  
	 \markboth{APPENDICE C}{APPENDICE C} \input{#2}}

\newcommand{\generaIndice}{
	%\fancyhead[L]{\textbf{\rightmark}}
	\newpage
	\tableofcontents
	\clearpage
	%\fancyhead[L]{\textbf{\leftmark}}
}

\newcommand{\generaIndiceTabelle}{\newpage \listoftables}

\newcommand{\generaIndiceFigure}{\newpage \listoffigures}

\let \OldBib \bibliography
\renewcommand{\bibliography}{
	\newpage 
	\markboth{Bibliografia}{Bibliografia} 
	\OldBib{biblios.bbl}}
\bibliographystyle{alpha}