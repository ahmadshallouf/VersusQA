import { Component } from '@angular/core';
import {AppStateService} from "../../../../StateManagement/app-state.service";

@Component({
  selector: 'app-arguments-presentation',
  templateUrl: './arguments-presentation.component.html',
  styleUrls: ['./arguments-presentation.component.css']
})
export class ArgumentsPresentationComponent {

      constructor(
          public state: AppStateService
      )
      { }

    // crop the domain out of url
    cropDomain(url: string): string {
        let v = url.split('/')[2];
        if (url.startsWith('http://')) {
            v = 'http://' + v;
        }else if (url.startsWith('https://')) {
            v = 'https://' + v;
        }
        return v;
    }

    // remove http, www and .* from url
    extractDomain (url: string): string {
        let domain = this.cropDomain(url);
        domain = domain.replace('http://', '');
        domain = domain.replace('https://', '');
        domain = domain.replace('www.', '');
        domain = domain.replace(/\/.*/, '');
        domain = domain.replace(/:.*/, '');
        domain = domain.split('.')[0];
        return domain;
    }
}
