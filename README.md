# CPS3320W01-Luna-Emma
Image features extraction and Image similarity comparison.

Project description:

Purpose: Perform Image features extraction; Judgement 3 kinds of similarities between two photos. 
1. Pixel-level similarity: Each corresponding pixel value of the two images is exactly equal, which is directly manifested by the fact that the two image files are identical in terms of binary content.
2. Global visual similarity: After two pixel-level similar images are scaled and compressed respectively, their corresponding pixel values also change to some extent due to scaling or compression, but remain visually identical.
3. Feature similarity: Determines whether the features are matching

Output: Print “T” or “F” for above similarity in each level. The interpretation of content we will print show in below rows:

Output;	Pixel-level;	Global visual;	Feature matching

① TTT:	Similar in Pixel-level;	Similar in Global visual;	Similar in Feature matching;

② FTT:	Dissimilar in Pixel-level;	Similar in Global visual;	Similar in Feature matching;

③ FFT:	Dissimilar in Pixel-level;	Dissimilar in Global visual;	Similar in Feature matching;

④ FFF:	Dissimilar in Pixel-level;	Dissimilar in Global visual;	Dissimilar in Feature matching;
