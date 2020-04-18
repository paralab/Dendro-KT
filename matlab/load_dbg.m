function [showA, showB] = load_dbg()
  cd "../build/_output";
  fineIm = fineRes_root();
  coarseIm = coarseRes_root();

  figFine = figure();
  figCoarse = figure();

  showA = @(n) myShowIt(n, figFine, fineIm)
  showB = @(n) myShowIt(n, figCoarse, coarseIm)

end

function myNewIm = myRescale(myIm)
  myNewIm = (myIm - min(myIm(:))) ./ (max(myIm(:)) - min(myIm(:)));
end

function myShowIt(n, f, i)
  figure(f);
  imshow(myRescale(i(:,:,n)));
end

